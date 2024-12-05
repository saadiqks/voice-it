import requests
import os
import wave
import re

from typing import List

NUM_CHANNELS = 1  # Assuming mono audio
SAMPLE_WIDTH = 2  # 2 bytes per sample (16-bit PCM)
SAMPLE_RATE = 16000  # The sample rate in Hz
MAX_LENGTH = 400


def chunk_text_by_sentences(text: str) -> List[str]:
   """
   Chunk text into segments, allowing sentence cutting if too long
   """
   # Split text into sentences using regex
   sentences = re.split(r'(?<=[.!?])\s*', text)

   chunks = []
   current_chunk = ""

   for sentence in sentences:
       # If sentence is longer than MAX_LENGTH, cut it
       if len(sentence) > MAX_LENGTH:
           # Cut sentence into MAX_LENGTH pieces
           while sentence:
               chunk = sentence[:MAX_LENGTH]
               chunks.append(chunk.strip())
               sentence = sentence[MAX_LENGTH:]
       else:
           # If adding this sentence would exceed MAX_LENGTH, start a new chunk
           if len(current_chunk) + len(sentence) > MAX_LENGTH:
               chunks.append(current_chunk.strip())
               current_chunk = sentence
           else:
               current_chunk += " " + sentence if current_chunk else sentence

   # Add the last chunk if not empty
   if current_chunk:
       chunks.append(current_chunk.strip())

   return chunks


def generate_speech_chunks(text: str) -> bytes:
    """
    Generate speech for text chunks.
    """
    api_key = os.getenv("LIGHTNING_API_TOKEN")

    text_chunks = chunk_text_by_sentences(text)
    combined_audio = b''

    url = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for chunk in text_chunks:
        payload = {
            "voice_id": "arman",
            "text": chunk,
            "sample_rate": SAMPLE_RATE
        }
        try:
            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                combined_audio += response.content
            else:
                print(f"Error processing chunk: {chunk}")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"Exception processing chunk: {e}")

    return combined_audio


def combine_wav_chunks(audio_data: bytes, output_path: str):
    """
    Combine multiple WAV file bytes into a single WAV file.
    """
    wav_path = output_path[:-3] + "wav"

    # Write the PCM data into a WAV file
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(NUM_CHANNELS)  # Mono channel
        wav_file.setsampwidth(SAMPLE_WIDTH)  # 16-bit audio (2 bytes)
        wav_file.setframerate(SAMPLE_RATE)  # SAMPLE RATE
        wav_file.writeframes(audio_data)  # Write the raw PCM data
