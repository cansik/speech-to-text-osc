import io
import pkgutil
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Optional, Union, Callable, Type, Dict

import faster_whisper
import numpy as np
import soundfile as sf
import speech_recognition as sr
import torch
import whisper
import whispercpp
from huggingface_hub import hf_hub_download

# patch whisper on file not find error
# https://github.com/carloscdias/whisper-cpp-python/pull/12
try:
    import whisper_cpp_python
except FileNotFoundError:
    regex = r"(\"darwin\":\n\s*lib_ext = \")\.so(\")"
    subst = "\\1.dylib\\2"

    print("fixing and re-importing whisper_cpp_python...")
    # load whisper_cpp_python and substitute .so with .dylib for darwin
    package = pkgutil.get_loader("whisper_cpp_python")
    whisper_path = Path(package.path)
    whisper_cpp_py = whisper_path.parent.joinpath("whisper_cpp.py")
    content = whisper_cpp_py.read_text()
    result = re.sub(regex, subst, content, 0, re.MULTILINE)
    whisper_cpp_py.write_text(result)

    import whisper_cpp_python


class WhisperModelType(Enum):
    Tiny = "tiny"
    Base = "base"
    Small = "small"
    Medium = "medium"
    Large = "large"
    LargeV2 = "large-v2"

    @staticmethod
    def from_text(model_name: str) -> "WhisperModelType":
        for model in WhisperModelType:
            if model.value == model_name:
                return model
        raise ValueError(f"No matching WhisperModelType for model name: {model_name}")


class WhisperBackend(ABC):

    @abstractmethod
    def __init__(self, model_name: str, device: str):
        pass

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        pass


WHISPER_BACKENDS: Dict[str, Type[WhisperBackend]] = {}


class OpenAIWhisperBackend(WhisperBackend):

    def __init__(self, model_name: str, device: str):
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        result = self.model.transcribe(audio_data, fp16=torch.cuda.is_available(), language=language)
        text = result['text'].strip()
        return text


WHISPER_BACKENDS["openai"] = OpenAIWhisperBackend


class FasterWhisperBackend(WhisperBackend):

    def __init__(self, model_name: str, device: str):
        if torch.cuda.is_available():
            self.model = faster_whisper.WhisperModel(model_name, device="cuda", compute_type="float16")
        else:
            self.model = faster_whisper.WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        segments, info = self.model.transcribe(audio_data, language=language, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text


WHISPER_BACKENDS["faster"] = FasterWhisperBackend


class WhisperCPPBackend(WhisperBackend):

    def __init__(self, model_name: str, device: str):
        model_path = Path(hf_hub_download(repo_id="ggerganov/whisper.cpp", filename=f"ggml-{model_name}.bin"))
        self.model: whispercpp.Whisper = object.__new__(whispercpp.Whisper)

        # init model manually because currently it is not possible to load a local models with the pip version 0.0.17
        # https://github.com/aarnphm/whispercpp/issues/126
        no_state = False
        self.model.context = whispercpp.api.Context.from_file(str(model_path), no_state=no_state)
        params = (  # noqa # type: ignore
            whispercpp.api.Params.from_enum(whispercpp.api.SAMPLING_GREEDY)
            .with_print_progress(False)
            .with_print_realtime(False)
            .build()
        )
        self.model.context.reset_timings()
        self.model._context_initialized = not no_state
        self.model._transcript = []
        self.model.__dict__.update(locals())

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        result = self.model.transcribe(audio_data)
        return result.strip()


WHISPER_BACKENDS["cpp"] = WhisperCPPBackend


class WhisperCPP2Backend(WhisperBackend):

    def __init__(self, model_name: str, device: str):
        model_path = Path(hf_hub_download(repo_id="ggerganov/whisper.cpp", filename=f"ggml-{model_name}.bin"))
        self.model = whisper_cpp_python.Whisper(str(model_path.absolute()))

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        # hacky solution to directly send numpy data
        self.model.params.language = language.encode('utf-8')
        self.model.params.temperature = 0.8
        result = self.model._full(audio_data)
        response = self.model._parse_format(result, "text")
        return response.strip()


WHISPER_BACKENDS["cpp2"] = WhisperCPP2Backend


@dataclass
class TextRecognitionEvent:
    index: int
    timestamp: datetime
    text: str
    inference_time: float = 0.0


class SpeechToTextWhisper:
    def __init__(self, model: WhisperModelType,
                 language: str = "en",
                 energy_threshold: float = 1000,
                 record_timeout: float = 2.0,
                 phrase_timeout: float = 3.0,
                 device: Optional[Union[str, torch.device]] = None,
                 whisper_backend: Type[WhisperBackend] = OpenAIWhisperBackend):
        self.phrase_start_ts: datetime = datetime.now()
        self.last_sample = bytes()
        self.data_queue = Queue()

        self.model = model

        self.whisper_backend = whisper_backend
        self.audio_model: Optional[WhisperBackend] = None

        self.device = device if device is not None else self._current_device()
        self.language = language

        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False

        self.source: Optional[sr.AudioFile, sr.Microphone] = None

        self._running = False

        self._last_sample = bytes()
        self._last_inference_time: float = 0
        self._transcription: str = ""
        self._transcription_id: int = -1
        self._transcription_start_ts: Optional[datetime] = None
        self._transcription_published: bool = True

        # events
        self.on_text_recognized: Optional[Callable[[TextRecognitionEvent], None]] = None
        self.on_partial_text_recognized: Optional[Callable[[TextRecognitionEvent], None]] = None

    def setup(self):
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

        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        # Read the transcription.
        text = self.audio_model.transcribe(audio_array, language=self.language)
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
                        TextRecognitionEvent(self._transcription_id, self._transcription_start_ts,
                                             self._transcription, self._last_inference_time)
                    )
            return

        if self.phrase_has_timed_out:
            # new transcription starts
            self._last_sample = bytes()
            self._transcription_published = False
            self._transcription_id += 1
            self._transcription_start_ts = now

        self.phrase_start_ts = now

        start_ts = time.time()
        text = self._transcribe_text()
        end_ts = time.time()

        self._last_inference_time = end_ts - start_ts

        self._transcription = text

        if self.on_partial_text_recognized is not None and text != "":
            self.on_partial_text_recognized(
                TextRecognitionEvent(self._transcription_id, self._transcription_start_ts,
                                     self._transcription, self._last_inference_time)
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

        if self.language == "en" and "large" not in model_name:
            model_name = f"{model_name}.en"

        self.audio_model = self.whisper_backend(model_name, self.device)

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
