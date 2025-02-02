import re
from io import StringIO
from typing import List

from config import TextConfig
from file_parsers import docx_to_string, pdf_to_string, txt_to_string


def get_text(file_path: str) -> str:
    text = ""
    file_type = file_path.split(".")[-1]

    match file_type:
        case "pdf":
            text = pdf_to_string(file_path)
        case "txt":
            text = txt_to_string(file_path)
        case "docx":
            text = docx_to_string(file_path)

    return text


def chunk_text_by_sentences(text: str) -> List[str]:
    sentence_list = re.split(r"(?<=[.!?])\s*", text)
    max_len = TextConfig.MAX_CHUNK_LENGTH
    chunk_list = []
    current_chunk = StringIO()

    for sentence in sentence_list:
        if len(sentence) > max_len:
            while sentence:
                chunk = sentence[:max_len]
                chunk_list.append(chunk.strip())
                sentence = sentence[max_len:]
        else:
            current_text = current_chunk.getvalue()
            space_needed = " " if current_text else ""

            if len(current_text) + len(space_needed) + len(sentence) > max_len:
                chunk_list.append(current_text.strip())
                current_chunk = StringIO()
                current_chunk.write(sentence)
            else:
                current_chunk.write(space_needed + sentence)

    final_text = current_chunk.getvalue()
    if final_text:
        chunk_list.append(final_text.strip())

    return chunk_list
