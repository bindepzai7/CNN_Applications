import os
import pandas as pd
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 👇 Adjust if your dataset path is different
DATA_PATH = "./data/ntc-scv/data/data_train/data_train/train"

def load_data(folder_path):
    examples = []
    for label in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                sentence = " ".join(line.strip() for line in f.readlines())
            label_numeric = 0 if label == "neg" else 1
            examples.append({"sentence": sentence, "label": label_numeric})
    return pd.DataFrame(examples)

print("📥 Loading training data...")
train_df = load_data(DATA_PATH)

# 👉 Import your preprocessing
from task3_preprocess import preprocess_text

# Apply text preprocessing
train_df["preprocess_sentence"] = train_df["sentence"].apply(preprocess_text)

# Initialize tokenizer
tokenizer = get_tokenizer("basic_english")

# Yield token generator
def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)

# Build vocabulary
print("📚 Building vocabulary...")
vocab = build_vocab_from_iterator(
    yield_tokens(train_df["preprocess_sentence"], tokenizer),
    max_tokens=10000,
    specials=["<pad>", "<unk>"]
)
vocab.set_default_index(vocab["<unk>"])

# Save vocab
os.makedirs("model", exist_ok=True)
with open("model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ Vocabulary saved to: model/vocab.pkl")
