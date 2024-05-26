# import spaces
from io import BytesIO
import numpy as np
import torch
from transformers.models.speecht5.number_normalizer import EnglishNumberNormalizer
from string import punctuation
import re
import soundfile as sf

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# repo_id = "parler-tts/parler_tts_mini_v0.1"
# repo_id = "models/parler_tts_mini_v0.1"
repo_id = "models/parler_tts_mini_v0.1_half"

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id)
model = model.eval()
# model = model.to(torch.bfloat16)
model = model.half()
# model.save_pretrained('models/parler_tts_mini_v0.1_half')
# torch.save(model, "model.bin")

model = model.to(device)

# save model to disk on safetensors

model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

SAMPLE_RATE = feature_extractor.sampling_rate
SEED = 42

default_text = "Please surprise me and speak in whatever voice you enjoy."
examples = [
    # [
    #     "Remember - this is only the first iteration of the model! To improve the prosody and naturalness of the speech further, we're scaling up the amount of training data by a factor of five times.",
    #     "A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a very clear audio and an animated tone."
    # ],
    # [
    #     "'This is the best time of my life, Bartley,' she said happily.",
    #     "A female speaker with a slightly low-pitched, quite monotone voice delivers her words at a slightly faster-than-average pace in a confined space with very clear audio.",
    # ],
    # [
    #     "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
    #     "A male speaker with a slightly high-pitched voice delivering his words at a slightly slow pace in a small, confined space with a touch of background noise and a quite monotone tone.",
    # ],
    # [
    #     "Montrose also, after having experienced still more variety of good and bad fortune, threw down his arms, and retired out of the kingdom.",
    #     "A male speaker with a low-pitched voice delivers his words at a fast pace and an animated tone, in a very spacious environment, accompanied by noticeable background noise.",
    # ],
]

number_normalizer = EnglishNumberNormalizer()


def preprocess(text):
    text = number_normalizer(text).strip()
    text = text.replace("-", " ")
    if text[-1] not in punctuation:
        text = f"{text}."

    abbreviations_pattern = r'\b[A-Z][A-Z\.]+\b'

    def separate_abb(chunk):
        chunk = chunk.replace(".", "")
        print(chunk)
        return " ".join(chunk)

    abbreviations = re.findall(abbreviations_pattern, text)
    for abv in abbreviations:
        if abv in text:
            text = text.replace(abv, separate_abb(abv))
    return text


def gen_tts(text, description):
    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(preprocess(text), return_tensors="pt").to(device)

    set_seed(SEED)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        generation = model.generate(
            input_ids=inputs.input_ids, prompt_input_ids=prompt.input_ids, do_sample=True, temperature=1.0
        )
    audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
    return SAMPLE_RATE, audio_arr


def write_wav(processed_np_speech, rate):
    # todo fix to use io.BytesIO
    bytes = BytesIO()
    bytes.name = "audio.wav"
    sf.write(bytes, processed_np_speech, rate, subtype='PCM_24')
    # bytesio to bytes
    return bytes.getvalue()


if __name__ == "__main__":
    start_time = time.time()
    rate, processed_np_speech = gen_tts(examples[0][0], examples[0][1])

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    wav = write_wav(processed_np_speech, rate)
    with open("audio.wav", "wb") as f:
        f.write(wav)
