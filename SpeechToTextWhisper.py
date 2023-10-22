import io
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Optional, Union, Callable

import speech_recognition as sr
import torch
import whisper
from whisper import Whisper


class WhisperModel(Enum):
    Tiny = "tiny"
    Base = "base"
    Small = "small"
    Medium = "medium"
    Large = "large"

    @staticmethod
    def from_text(model_name: str) -> "WhisperModel":
        for model in WhisperModel:
            if model.value == model_name:
                return model
        raise ValueError(f"No matching WhisperModel for model name: {model_name}")


@dataclass
class TextRecognitionEvent:
    index: int
    timestamp: datetime
    text: str


class SpeechToTextWhisper:
    def __init__(self, model: WhisperModel,
                 language: str = "en",
                 energy_threshold: float = 1000,
                 record_timeout: float = 2.0,
                 phrase_timeout: float = 3.0,
                 device: Optional[Union[str, torch.device]] = None):
        self.phrase_start_ts: datetime = datetime.now()
        self.last_sample = bytes()
        self.data_queue = Queue()

        self.model = model
        self.audio_model: Optional[Whisper] = None
        self.device = device if device is not None else self._current_device()
        self.language = language

        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False

        self.source: Optional[sr.AudioFile, sr.Microphone] = None

        self.temp_file = NamedTemporaryFile().name

        self._running = False

        self._last_sample = bytes()
        self._transcription: str = ""
        self._transcription_id: int = -1
        self._transcription_start_ts: Optional[datetime] = None
        self._transcription_published: bool = True

        # events
        self.on_text_recognized: Optional[Callable[[TextRecognitionEvent], None]] = None
        self.on_partial_text_recognized: Optional[Callable[[TextRecognitionEvent], None]] = None

    def setup(self):
        self.temp_file = NamedTemporaryFile().name

        self._load_model()
        self._init_audio()

    def run(self):
        """
        Run text to speech recognition. Blocks until canceled.
        """
        self._running = True

        # init variables
        self._last_sample = bytes()

        # run loop (blocking)
        while self._running:
            try:
                self._analyze_audio()
                time.sleep(0.1)
            except KeyboardInterrupt:
                self._running = False

    def stop(self):
        self._running = False

    def _transcribe_text(self) -> str:
        while not self.data_queue.empty():
            data = self.data_queue.get()
            self._last_sample += data

        # Use AudioData to convert the raw data to wav data.
        audio_data = sr.AudioData(self._last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
        wav_data = io.BytesIO(audio_data.get_wav_data())

        # Write wav data to the temporary file as bytes.
        with open(self.temp_file, 'w+b') as f:
            f.write(wav_data.read())

        # Read the transcription.
        result = self.audio_model.transcribe(self.temp_file, fp16=torch.cuda.is_available(), language=self.language)
        text = result['text'].strip()
        return text

    def _analyze_audio(self):
        now = datetime.now()

        # check if new data has arrived
        if self.data_queue.empty():
            # fire text-detection event
            if self.phrase_has_timed_out and not self._transcription_published:
                self._transcription_published = True
                if self.on_text_recognized is not None and self._transcription != "":
                    self.on_text_recognized(
                        TextRecognitionEvent(self._transcription_id, self._transcription_start_ts, self._transcription)
                    )
            return

        if self.phrase_has_timed_out:
            # new transcription starts
            self._last_sample = bytes()
            self._transcription_published = False
            self._transcription_id += 1
            self._transcription_start_ts = now

        self.phrase_start_ts = now

        text = self._transcribe_text()
        self._transcription = text

        if self.on_partial_text_recognized is not None and text != "":
            self.on_partial_text_recognized(
                TextRecognitionEvent(self._transcription_id, self._transcription_start_ts, self._transcription)
            )

    def _init_audio(self):
        self.source = sr.Microphone(sample_rate=16000)

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        def record_callback(_, audio: sr.AudioData) -> None:
            """
            Threaded callback function to receive audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            data = audio.get_raw_data()
            self.data_queue.put(data)

        self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=self.record_timeout)

    def _load_model(self) -> None:
        model_name: str = self.model.value

        if self.language == "en" and model_name != "large":
            model_name = f"{model_name}.en"

        self.audio_model = whisper.load_model(model_name, device=self.device)

    @staticmethod
    def _current_device() -> torch.device:
        # mps currently not available
        if torch.backends.mps.is_available() and False:
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    @property
    def phrase_has_timed_out(self) -> bool:
        now = datetime.now()
        return now - self.phrase_start_ts > timedelta(seconds=self.phrase_timeout)
