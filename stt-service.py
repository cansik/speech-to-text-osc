import argparse

from pythonosc import udp_client
from rich.console import Console

from SpeechToTextWhisper import SpeechToTextWhisper, WhisperModel, TextRecognitionEvent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tiny", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2"])
    parser.add_argument("--language", type=str, default="en",
                        help="Language code to decode (ISO 639-1 format).")
    parser.add_argument("--faster", action="store_true", help="Use faster-whisper audio transcription.")

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

    osc_client = udp_client.SimpleUDPClient(args.osc_server, args.osc_port)

    def on_partial_text_recognized(event: TextRecognitionEvent):
        osc_client.send_message("/stt/partial-text",
                                [event.index, event.timestamp.isoformat(), event.text])
        console.print(f"{event.text}", style="blue")

    def on_text_recognized(event: TextRecognitionEvent):
        osc_client.send_message("/stt/text",
                                [event.index, event.timestamp.isoformat(), event.text])
        ts_str = event.timestamp.strftime("%H:%M:%S")
        console.print(f"{ts_str}: {event.text}", style="green bold")

    model_type = "faster-whisper" if args.faster else "whisper"
    with console.status(f"starting {model_type} model {args.model} for language {args.language}"):
        stt_transcriber = SpeechToTextWhisper(model=WhisperModel.from_text(args.model),
                                              language=args.language,
                                              energy_threshold=args.energy_threshold,
                                              record_timeout=args.record_timeout,
                                              phrase_timeout=args.phrase_timeout,
                                              use_faster_whisper=args.faster)

        stt_transcriber.setup()

    stt_transcriber.on_text_recognized = on_text_recognized
    stt_transcriber.on_partial_text_recognized = on_partial_text_recognized

    console.print("listening...")
    stt_transcriber.run()


if __name__ == "__main__":
    main()
