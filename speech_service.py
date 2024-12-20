from config import SpeechConfig
from google.cloud import texttospeech
from google.cloud.texttospeech import SynthesizeSpeechResponse


def generate_speech(text: str) -> SynthesizeSpeechResponse:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=SpeechConfig.NAME
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response


def combine_wav_chunks(audio_data: SynthesizeSpeechResponse, output_path: str) -> None:
    with open(output_path, "wb") as out:
        out.write(audio_data.audio_content)
