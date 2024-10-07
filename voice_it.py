import html2text
import os
import soundfile as sf
import torch

from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
from threading import Thread

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from docx import Document

def streaming_text_to_audio():
    device = "cpu"
    torch_dtype = torch.bfloat16
    model_name = "parler-tts/parler-tts-mini-v1"

    # Need to set padding max length
    max_length = 50

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
    ).to(device, dtype=torch_dtype)

    sampling_rate = model.audio_encoder.config.sampling_rate
    frame_rate = model.audio_encoder.config.frame_rate

    def generate(text, description, play_steps_in_s=0.5):
        play_steps = int(frame_rate * play_steps_in_s)
        streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)
        # Tokenization
        inputs = tokenizer(description, return_tensors="pt").to(device)
        prompt = tokenizer(text, return_tensors="pt").to(device)
        # Create generation kwargs
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
        # Initialize Thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        # Iterate over chunks of audio
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
            yield sampling_rate, new_audio


    text = "This is a test of the streamer class"
    description = "Jon's talking really fast."

    chunk_size_in_s = 0.5

    for (sampling_rate, audio_chunk) in generate(text, description, chunk_size_in_s):
    # You can do everything that you need with the chunk now
    # For example: stream it, save it, play it.
        print(audio_chunk.shape)

def convert_text_to_speech(text: str, mp3_path: str) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "parler-tts/parler-tts-mini-v1"

    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    # prompt = "the quick brown fox"
    prompt = text[:100]
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    generation = model.generate(input_ids=input_ids, attention_mask=attention_mask, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(mp3_path, audio_arr, model.config.sampling_rate)

def txt_to_text(file_path: str) -> str:
    with open(file_path, "r") as file:
        text = file.read()

    return text

def epub_to_text(file_path: str) -> str:
    book = epub.read_epub(file_path)
    content = ""

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            html_content = item.get_content().decode("utf-8")
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text()
            content += text + "\n\n"

    h = html2text.HTML2Text()
    h.ignore_links = True
    content = h.handle(content)

    return content.strip()

def docx_to_text(file_path: str) -> str:
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]

    return "\n".join(full_text)

def pdf_to_text(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def convert_file_to_mp3(file_path: str, mp3_path: str) -> None:
    text = ""
    file_type = file_path.split(".")[-1]

    match file_type:
        case "pdf":
            text = pdf_to_text(file_path)
        case "txt":
            text = txt_to_text(file_path)
        case "epub":
            text = epub_to_text(file_path)
        case "docx":
            text = docx_to_text(file_path)

    convert_text_to_speech(text, mp3_path)
    print(f"Conversion complete. Audio saved to {mp3_path}.")
    # streaming_text_to_audio()
