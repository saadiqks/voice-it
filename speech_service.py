import requests
import wave

from text_service import chunk_text_by_sentences
from config import SpeechConfig


def generate_speech_chunks(text: str) -> bytes:
    text_chunks = chunk_text_by_sentences(text)
    combined_audio = b''

    headers = {
        "Authorization": f"Bearer {SpeechConfig.API_TOKEN}",
        "Content-Type": "application/json"
    }

    for chunk in text_chunks:
        payload = {
            "voice_id": "arman",
            "text": chunk,
            "sample_rate": SpeechConfig.SAMPLE_RATE
        }
        try:
            response = requests.post(SpeechConfig.API_ENDPOINT, json=payload, headers=headers)

            if response.status_code == 200:
                combined_audio += response.content
            else:
                print(f"Error processing chunk: {chunk}")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"Exception processing chunk: {e}")

    return combined_audio


def combine_wav_chunks(audio_data: bytes, output_path: str) -> None:
    # Write the PCM data into a WAV file
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(SpeechConfig.NUM_CHANNELS)
        wav_file.setsampwidth(SpeechConfig.SAMPLE_WIDTH)
        wav_file.setframerate(SpeechConfig.SAMPLE_RATE)
        wav_file.writeframes(audio_data)
