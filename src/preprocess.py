import re
from collections import Counter
from typing import List, Tuple, Dict

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


class Vocabulary:
    def __init__(self, max_size: int = 20000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts: List[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        idx = 2
        for word, freq in sorted_items:
            if freq < self.min_freq:
                continue
            if idx >= self.max_size:
                break
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def encode(self, text: str, max_len: int = 200) -> Tuple[List[int], int]:
        tokens = tokenize(text)
        length = min(len(tokens), max_len)
        ids = [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.stoi[PAD_TOKEN]] * (max_len - len(ids))
        return ids, length

    def __len__(self) -> int:
        return len(self.stoi)
