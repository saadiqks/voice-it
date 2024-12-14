import re
from typing import List
from file_parsers import pdf_to_string, txt_to_string, docx_to_string
from config import TextConfig

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
   sentence_list = re.split(r'(?<=[.!?])\s*', text)
   max_len = TextConfig.MAX_CHUNK_LENGTH

   chunk_list = []
   current_chunk = ""

   for sentence in sentence_list:
       if len(sentence) > max_len:
           while sentence:
               chunk = sentence[:max_len]
               chunk_list.append(chunk.strip())
               sentence = sentence[max_len:]
       else:
           if len(current_chunk) + len(sentence) > max_len:
               chunk_list.append(current_chunk.strip())
               current_chunk = sentence
           else:
               current_chunk += " " + sentence if current_chunk else sentence

   if current_chunk:
       chunk_list.append(current_chunk.strip())

   return chunk_list
