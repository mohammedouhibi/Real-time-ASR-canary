from vad import VadInferencer
import pyaudio
import threading
import time
from time import monotonic
import math
from collections import deque

import numpy as np
import os
import torch
from ruamel.yaml import YAML
from omegaconf import DictConfig
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
import onnxruntime
from nemo.collections.asr.metrics import wer

import soxr
import librosa


class QuartznetInferencer():
    def __init__(self):

        self.stt_config_path = "model/quartznet5x5.yaml"

        yaml = YAML(typ='safe')
        with open(self.stt_config_path, encoding="utf-8") as f:
            self.params = yaml.load(f)

        self.load_onnx_model()

    def load_onnx_model(self):

        model_to_load = "model/stt_model.onnx"

        # create preprocessor
        self.preprocessor = EncDecCTCModel.from_config_dict(DictConfig(self.params['model']).preprocessor)

        self.sess = onnxruntime.InferenceSession(model_to_load)

        self.input_name = self.sess.get_inputs()[0].name
        self.label_name = self.sess.get_outputs()[0].name

        runs = 5
        input_data = np.zeros((1, 64, 352), np.float32) #3rd variable changes according to input audio
        # Warming up
        for i in range(runs):
            _ = self.sess.run([], {self.input_name: input_data})

        self.wer = wer.WER(
            vocabulary=self.params["labels"],
            batch_dim_index=0,
            use_cer=False,
            ctc_decode=True,
            dist_sync_on_step=True,
        )
        print("Onnx model loaded!")

    def inference(self, data):

        signal = np.expand_dims(data, 0)  # add a batch dimension
        signal = torch.from_numpy(signal)  # converts the NumPy array to a PyTorch tensor

        processed_signal, _ = self.preprocessor(input_signal=signal,
                                                length=torch.tensor(data.size).unsqueeze(0),)
        processed_signal = processed_signal.cpu().numpy()

        # inference
        logits = self.sess.run([self.label_name], {self.input_name: processed_signal})
        probabilities = logits[0][0]
        a = np.array([np.argmax(x) for x in probabilities])
        a = np.expand_dims(a, 0)
        a = torch.from_numpy(a)
        prediction = self.wer.ctc_decoder_predictions_tensor(a)

        return prediction[0]

class VoiceListener:

    def __init__(self):

        self.listening_thread = None
        self.silent_time = None
        self.has_started_speaking = False
        self.has_finished_speaking = False

        # listener parameters
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 48000

        self.MAX_WAIT = 1.0
        self.vad_frame_queue = deque(maxlen=8)

        self.vad = VadInferencer()
        self.setup_audio_stream()

    def setup_audio_stream(self):
        p = pyaudio.PyAudio()



        self.frames = []
        self.mode = "listen"

        # Define audio callback function
        self.stream = p.open(format=self.FORMAT,
                             channels=self.CHANNELS,
                             rate=self.RATE,
                             input=True,
                             frames_per_buffer=self.CHUNK)

    def start_listening(self):
        self.listening_thread = threading.Thread(target=self.listener, args=(), daemon=True)
        self.listening_thread.start()

    def listener(self):
        while True:
            self.stream_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            # if self.has_started_speaking:
            if self.mode == "listen":
                self.frames.append(self.stream_data)
            self.vad_frame_queue.append(self.stream_data)

    def stop_listening(self):
        self.has_finished_speaking = False
        self.has_started_speaking = False

    def vad_(self):

        if len(self.vad_frame_queue) >= 8:
            bytes = b''
            for frame in self.vad_frame_queue:
                bytes += frame
            self.vad_frame_queue.clear()
            if self.vad.has_speech_activity(audio_chunk=bytes):
                self.silent_time = None

                if not self.has_started_speaking:
                    self.mode = "listen"
                    self.has_started_speaking = True
                    print("Speech activity detected!")

            elif self.has_started_speaking:
                if self.silent_time is None:
                    self.silent_time = monotonic()
                elif monotonic() - self.silent_time > self.MAX_WAIT:
                    self.has_finished_speaking = True
                    self.mode = "wait"
                    print("Speech has ended!")


    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound


if __name__ == '__main__':

    quartznet_inferencer = QuartznetInferencer()
    voice_listener = VoiceListener()

    print("\nWaiting for speech...\n")
    voice_listener.start_listening()

    while True:
        voice_listener.vad_()

        if voice_listener.mode == "wait":

            #  take real time audio bytes
            stt_bytes = b''
            for chunk in voice_listener.frames:
                stt_bytes += chunk

            # convert bytes to numpy array
            audio_int16 = np.fromstring(stt_bytes, dtype=np.int16)
            # convert array int6 --> float32
            audio_float32 = voice_listener.int2float(audio_int16)
            # resample audio array to 16khZ since model inference can be run at 16kZ
            # if your mic device support 16khZ, you do not need to apply resample.
            audio_data = soxr.resample(audio_float32, 48000, 16000, quality=soxr.VHQ)
            start = time.time()
            prediction = quartznet_inferencer.inference(audio_data)
            print("\nPrediction:", prediction)
            print("Inference time:", time.time() - start)

            voice_listener.frames = []
            voice_listener.stop_listening()

            voice_listener.mode = "listen"
            print("waiting for speech...")



