import os
import pandas as pd
from langid.langid import LanguageIdentifier, model
import time
import random
import re
import string
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def load_data(folder_path):
    examples = []

    for label in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                sentence = " ".join(line.strip() for line in lines)

            # Convert label string to numeric
            label_numeric = 0 if label == "neg" else 1 if label == "pos" else None

            examples.append({
                "sentence": sentence,
                "label": label_numeric
            })

    return pd.DataFrame(examples)


def identify_vn(df):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    not_vi_idx = set()
    THRESHOLD = 0.9

    for idx, row in df.iterrows():
        sentence = row["sentence"]
        lang, prob = identifier.classify(sentence)
        if lang != "vi" or prob <= THRESHOLD:
            not_vi_idx.add(idx)

    vi_df = df[~df.index.isin(not_vi_idx)]
    not_vi_df = df[df.index.isin(not_vi_idx)]

    return vi_df, not_vi_df

def preprocess_text(text):
    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(" ", text)

    # Remove HTML tags
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # Remove punctuation and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(" ", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.lower()

def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)
        