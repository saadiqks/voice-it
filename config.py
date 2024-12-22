from dotenv import load_dotenv

load_dotenv()

class SpeechConfig:
    NAME = "en-US-Wavenet-J"

class FileConfig:
    ALLOWED_MIME_TYPES = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    UPLOAD_FOLDER = "/tmp"

class TextConfig:
    MAX_TEXT_LENGTH = 2000 # Google TTS has a limit of 5000 chars
    MAX_CHUNK_LENGTH = 400
