from speech_service import generate_speech_chunks, combine_wav_chunks
from text_service import get_text
from config import TextConfig
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from magic import Magic
import os
from config import FileConfig


def validate_mime_type(file_path: str) -> bool:
    mime = Magic(mime=True)
    file_mime_type = mime.from_file(file_path)

    return file_mime_type in FileConfig.ALLOWED_MIME_TYPES


def save_uploaded_file(file: FileStorage) -> None | str:
    if not file or file.filename is None or file.filename == "":
        return None

    filename = secure_filename(file.filename)
    file_path = os.path.join(FileConfig.UPLOAD_FOLDER, filename)
    file.save(file_path)

    if not validate_mime_type(file_path):
        os.remove(file_path)
        return None

    return file_path


def convert_file_to_wav(file_path: str, audio_path: str) -> None | int:
    text = get_text(file_path)
    text_len = len(text)
    if text_len <= TextConfig.MAX_TEXT_LENGTH:
        audio_data = generate_speech_chunks(text)
        combine_wav_chunks(audio_data, audio_path)
    else:
        return text_len
