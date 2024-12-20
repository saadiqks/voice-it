import os

from dotenv import load_dotenv

load_dotenv()

class SpeechConfig:
    NAME = "en-US-Wavenet-J"

class FileConfig:
    ALLOWED_MIME_TYPES = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingtml.document"
    ]

    UPLOAD_FOLDER = "/tmp"

class TextConfig:
    MAX_TEXT_LENGTH = 5000 # Google TTS has a limit of 5000 chars
    MAX_CHUNK_LENGTH = 400

class LoggingConfig:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(os.getenv("LOG_DIR", "/var/log"), "voice_it.log")

class FeatureFlags:
    CHUNK_PROCESSING_ENABLED = True
    CACHING_ENABLED = False
