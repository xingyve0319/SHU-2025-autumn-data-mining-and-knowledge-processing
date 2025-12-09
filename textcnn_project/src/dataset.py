import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    labels = df[0].values - 1  # 转成0/1
    texts = (df[1] + " " + df[2]).apply(clean_text).tolist()
    return texts, labels

def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {word: idx + 2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def encode_text(text, vocab, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        label = self.labels[idx]
        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
