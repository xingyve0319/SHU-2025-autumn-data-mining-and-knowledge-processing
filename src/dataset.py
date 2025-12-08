import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset
from src.utils import clean_text, tokenize

def load_csv_no_header(path):
    # CSV 行结构：label, title, text
    df = pd.read_csv(path, header=None)
    df.columns = ["label", "title", "text"]
    return df

def build_vocab(texts, vocab_size=50000, min_freq=1):
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)

    words = [w for w, c in counter.items() if c >= min_freq]
    words = sorted(words, key=lambda x: counter[x], reverse=True)[:vocab_size]

    stoi = {w: i+2 for i, w in enumerate(words)}  # 0=PAD, 1=UNK
    stoi["<PAD>"] = 0
    stoi["<UNK>"] = 1

    return stoi

def encode(tokens, stoi, max_len):
    ids = [stoi.get(w, 1) for w in tokens]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))

class TextDataset(Dataset):
    def __init__(self, df, stoi, max_len=200):
        self.stoi = stoi
        self.max_len = max_len
        self.texts = []
        self.labels = []

        for _, row in df.iterrows():
            text = clean_text(row["text"])
            tokens = tokenize(text)
            token_ids = encode(tokens, stoi, max_len)
            self.texts.append(token_ids)

            # label: 1=neg, 2=pos  → 转为 0/1
            label = int(row["label"])
            self.labels.append(1 if label == 2 else 0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.texts[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
