import os
from dotenv import load_dotenv

load_dotenv()

class SpeechConfig:
    API_TOKEN = os.getenv("API_TOKEN")
    API_ENDPOINT = "https://waves-api.smallest.ai/api/v1/lightning/get_speech"

    NUM_CHANNELS = 1  # Mono audio
    SAMPLE_WIDTH = 2  # 2 bytes per sample (16-bit PCM)
    SAMPLE_RATE = 24000  # Sample rate in Hz

class FileConfig:
    ALLOWED_MIME_TYPES = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingtml.document"
    ]

    UPLOAD_FOLDER = "/tmp"

class TextConfig:
    MAX_TEXT_LENGTH = 3000
    MAX_CHUNK_LENGTH = 400

class LoggingConfig:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(os.getenv("LOG_DIR", "/var/log"), "voice_it.log")

class FeatureFlags:
    CHUNK_PROCESSING_ENABLED = True
    CACHING_ENABLED = False
