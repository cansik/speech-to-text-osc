import argparse

import pyaudio
from pythonosc import udp_client
from rich.console import Console

from SpeechToTextWhisper import SpeechToTextWhisper, WhisperModelType, TextRecognitionEvent, WHISPER_BACKENDS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=WhisperModelType.Base.value, help="Model to use.",
                        choices=[t.value for t in WhisperModelType])
    parser.add_argument("--language", type=str, default="en",
                        help="Language code to decode (ISO 639-1 format).")
    parser.add_argument("--backend", type=str, default=list(WHISPER_BACKENDS.keys())[0],
                        help="Backend to use.", choices=[k for k in WHISPER_BACKENDS.keys()])

    # find current devices
    context = pyaudio.PyAudio()
    devices = {i: context.get_device_info_by_index(i)["name"] for i in range(context.get_device_count())}
    microphone_names = ", ".join([f"{i}={n}" for i, n in devices.items()])

    parser.add_argument("--audio-device", type=int, default=None,
                        help=f"Audio input id, one of [{microphone_names}].")
    parser.add_argument("--energy-threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record-timeout", default=0.5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase-timeout", default=1.2,
                        help="How much empty space between recordings before the message is sent", type=float)

    parser.add_argument("--osc-server", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument("--osc-port", type=int, default=8000, help="OSC output port.")

    return parser.parse_args()


def main():
    console = Console()
    args = parse_arguments()

    backend_type = WHISPER_BACKENDS[args.backend]

    osc_client = udp_client.SimpleUDPClient(args.osc_server, args.osc_port)

    def on_partial_text_recognized(event: TextRecognitionEvent):
        osc_client.send_message("/stt/partial-text",
                                [event.index, event.timestamp.isoformat(), event.text])

        inference_time_ms = round(event.inference_time * 1000)
        console.print(f"[italic]{inference_time_ms}ms:[/italic] {event.text}", style="blue")

    def on_text_recognized(event: TextRecognitionEvent):
        osc_client.send_message("/stt/text",
                                [event.index, event.timestamp.isoformat(), event.text])
        ts_str = event.timestamp.strftime("%H:%M:%S")
        console.print(f"{ts_str}: {event.text}", style="green bold")

    with console.status(f"starting whisper model {args.model} ({args.backend}) for language {args.language}"):
        stt_transcriber = SpeechToTextWhisper(model=WhisperModelType.from_text(args.model),
                                              language=args.language,
                                              energy_threshold=args.energy_threshold,
                                              record_timeout=args.record_timeout,
                                              phrase_timeout=args.phrase_timeout,
                                              whisper_backend=backend_type,
                                              audio_device_index=args.audio_device)

        stt_transcriber.setup()

    stt_transcriber.on_text_recognized = on_text_recognized
    stt_transcriber.on_partial_text_recognized = on_partial_text_recognized

    console.print("listening...")
    stt_transcriber.run()


if __name__ == "__main__":
    main()
