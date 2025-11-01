# preprocess.py
"""
Preprocessing utilities for COCO image captioning project.

Provides:
- Vocabulary: word <-> index mapping with special tokens,
  methods: add_word, numericalize, save, load
- build_from_coco(ann_file, freq_threshold): build vocab from COCO captions JSON
- get_transforms(train=True, image_size=224): torchvision transforms for train/val
"""

import os
import re
import json
import pickle
from collections import Counter
from typing import List

import nltk
# download punkt once (quiet)
nltk.download("punkt", quiet=True)

from torchvision import transforms

class Vocabulary:
    # canonical token names
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    # short alias names (used by some other files)
    PAD = PAD_TOKEN
    START = START_TOKEN
    END = END_TOKEN
    UNK = UNK_TOKEN

    def __init__(self, freq_threshold:int = 5):
        """
        freq_threshold: when building from captions, minimum frequency to include a word.
        """
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()
        self._next_index = 0
        # initialize special tokens
        for tok in (self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN):
            self.add_word(tok)
        # convenience property
        self.pad_index = self.word2idx[self.PAD_TOKEN]
        self.start_index = self.word2idx[self.START_TOKEN]
        self.end_index = self.word2idx[self.END_TOKEN]
        self.unk_index = self.word2idx[self.UNK_TOKEN]

    def add_word(self, word: str):
        """Add a single word to the vocab (if not present)."""
        if word not in self.word2idx:
            self.word2idx[word] = self._next_index
            self.idx2word[self._next_index] = word
            self._next_index += 1

    def __call__(self, word: str) -> int:
        """Return index for word (UNK if missing)."""
        return self.word2idx.get(word, self.unk_index)

    def __len__(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a string using NLTK (fallback to simple regex)."""
        try:
            tokens = nltk.tokenize.word_tokenize(text.lower())
            if tokens:
                return tokens
        except Exception:
            pass
        # fallback
        return re.findall(r"\w+", text.lower())

    def build_from_coco(self, ann_file: str):
        """
        Build vocabulary frequencies from a COCO captions JSON file.
        After calling this, add words above freq_threshold to the vocab.
        """
        assert os.path.isfile(ann_file), f"Annotation file not found: {ann_file}"
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # count tokens across all annotations
        for i, ann in enumerate(coco.get("annotations", []), 1):
            caption = ann.get("caption", "")
            tokens = self.tokenize(caption)
            self.freqs.update(tokens)
            if i % 1000 == 0:
                print(f"[{i}] processed captions...")

        # add words meeting threshold
        added = 0
        for word, freq in self.freqs.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.add_word(word)
                added += 1
        print(f"Added {added} words to vocab (threshold={self.freq_threshold}). Total vocab size: {len(self)}")

    def numericalize(self, caption: str) -> List[int]:
        """
        Convert a caption string to list of indices:
        [<start>, w1, w2, ..., <end>]
        """
        tokens = self.tokenize(caption)
        ids = [self.start_index]
        for t in tokens:
            ids.append(self.word2idx.get(t, self.unk_index))
        ids.append(self.end_index)
        return ids

    def save(self, path: str):
        """Save vocabulary object to a pickle file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
                "freqs": self.freqs,
                "freq_threshold": self.freq_threshold,
                "next_index": self._next_index
            }, f)
        print(f"Saved vocabulary to {path}")

    @staticmethod
    def load(path: str) -> "Vocabulary":
        """Load a vocabulary saved with `.save()`."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = Vocabulary(freq_threshold=data.get("freq_threshold", 5))
        # reset and restore state
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        vocab.freqs = data.get("freqs", Counter())
        vocab._next_index = data.get("next_index", max(vocab.idx2word.keys()) + 1 if vocab.idx2word else len(vocab.word2idx))
        # rebuild convenience indices
        vocab.pad_index = vocab.word2idx.get(Vocabulary.PAD_TOKEN)
        vocab.start_index = vocab.word2idx.get(Vocabulary.START_TOKEN)
        vocab.end_index = vocab.word2idx.get(Vocabulary.END_TOKEN)
        vocab.unk_index = vocab.word2idx.get(Vocabulary.UNK_TOKEN)
        return vocab


def get_transforms(train: bool = True, image_size: int = 224):
    """
    Return torchvision transforms.
    - train=True: augmentation (random crop + hflip)
    - train=False: center crop (deterministic)
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])


if __name__ == "__main__":
    # Example usage:
    # python preprocess.py --caption_path data/coco/annotations_trainval2014/annotations/captions_train2014.json --vocab_path preprocessed/vocab.pkl --threshold 5
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path", type=str, help="Path to COCO captions json", required=False)
    parser.add_argument("--vocab_path", type=str, default="preprocessed/vocab.pkl", help="Where to save the vocab")
    parser.add_argument("--threshold", type=int, default=5, help="Minimum token frequency")
    args = parser.parse_args()

    if args.caption_path:
        v = Vocabulary(freq_threshold=args.threshold)
        v.build_from_coco(args.caption_path)
        v.save(args.vocab_path)
    else:
        print("No caption_path provided. This module contains Vocabulary and get_transforms utilities.")
