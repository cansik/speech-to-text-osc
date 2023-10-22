import argparse
import io
import os
import re
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

from rich.console import Console

from SpeechToTextWhisper import SpeechToTextWhisper, WhisperModel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", type=str, default="en",
                        help="Language code to decode (ISO 639-1 format).")

    parser.add_argument("--energy-threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record-timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase-timeout", default=3,
                        help="How much empty space between recordings before the message is sent", type=float)

    return parser.parse_args()


def main():
    args = parse_arguments()

    console = Console()
    with console.status("loading model"):
        stt_transcriber = SpeechToTextWhisper(model=WhisperModel.from_text(args.model),
                                              language=args.language,
                                              energy_threshold=args.energy_threshold,
                                              record_timeout=args.record_timeout,
                                              phrase_timeout=args.phrase_timeout)

        stt_transcriber.setup()

    stt_transcriber.run()


if __name__ == "__main__":
    main()
