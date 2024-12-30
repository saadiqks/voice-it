from docx import Document
from PyPDF2 import PdfReader
from io import StringIO


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
        ss = StringIO()

        for page in reader.pages:
            ss.write(f"{page.extract_text()}\n")

    return ss.getvalue()
