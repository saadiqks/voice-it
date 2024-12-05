from docx import Document
from PyPDF2 import PdfReader
from utils import generate_speech_chunks, combine_wav_chunks

CHAR_LIMIT = 1500

def txt_to_string(file_path: str) -> str:
    with open(file_path, "r") as file:
        text = file.read()
    return text


def docx_to_string(file_path: str) -> str:
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]

    return "\n".join(full_text)


def pdf_to_string(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def call_lightning_api(text, audio_path):
    audio_data = generate_speech_chunks(text)
    combine_wav_chunks(audio_data, audio_path)


def convert_file_to_wav(file_path: str, audio_path: str) -> None:
    text = ""
    file_type = file_path.split(".")[-1]

    match file_type:
        case "pdf":
            text = pdf_to_string(file_path)
        case "txt":
            text = txt_to_string(file_path)
        case "docx":
            text = docx_to_string(file_path)

    if len(text) > CHAR_LIMIT:
        raise Exception("File has too many characters.")

    call_lightning_api(text, audio_path)
