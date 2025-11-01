# data_loader.py
"""
Classic COCO captions DataLoader.

- COCODataset: yields (image_tensor, caption_tensor)
- collate_fn: pads captions in a batch
- get_loader: convenience to get a torch.utils.data.DataLoader
"""
import os
import json
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from preprocess import Vocabulary, get_transforms  # local import (same folder)

class COCODataset(Dataset):
    def __init__(self, images_root: str, ann_file: str, vocab: Vocabulary, transform=None, max_len: int = 50):
        assert os.path.isdir(images_root), f"Images folder not found: {images_root}"
        assert os.path.isfile(ann_file), f"Annotation file not found: {ann_file}"

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # map id -> filename
        id2file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

        self.samples = []
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            fname = id2file.get(img_id)
            if not fname:
                continue
            img_path = os.path.join(images_root, fname)
            if not os.path.isfile(img_path):
                continue
            self.samples.append((img_path, ann["caption"]))

        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, caption = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caps_ids = self.vocab.numericalize(caption)
        if len(caps_ids) > self.max_len:
            caps_ids = caps_ids[:self.max_len-1] + [self.vocab.word2idx[self.vocab.END]]
        cap_tensor = torch.tensor(caps_ids, dtype=torch.long)
        return img, cap_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)  # PAD id = 0
    return images, captions_padded, lengths


def get_loader(images_root: str, ann_file: str, vocab: Vocabulary, batch_size: int = 32, shuffle: bool = True,
               num_workers: int = 0, train: bool = True):
    transforms = get_transforms(train=train)
    dataset = COCODataset(images_root, ann_file, vocab, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return loader


if __name__ == "__main__":
    # Example run (adjust paths to your project):
    # python data_loader.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", type=str, default="data/coco/train2014/train2014",
                        help="path to images folder (e.g. data/coco/train2014/train2014)")
    parser.add_argument("--ann_file", type=str, default="data/coco/annotations_trainval2014/annotations/captions_train2014.json")
    parser.add_argument("--vocab_path", type=str, default="preprocessed/vocab.pkl")
    parser.add_argument("--build_vocab", action="store_true")
    args = parser.parse_args()

    if args.build_vocab or not os.path.isfile(args.vocab_path):
        vocab = Vocabulary(freq_threshold=5)
        vocab.build_from_coco(args.ann_file)
        vocab.save(args.vocab_path)
    else:
        vocab = Vocabulary.load(args.vocab_path)

    loader = get_loader(args.images_root, args.ann_file, vocab, batch_size=8, num_workers=0, train=True)
    print("Vocab size:", len(vocab))
    for i, (images, captions, lengths) in enumerate(loader):
        print(i, images.shape, captions.shape, lengths[:6])
        if i >= 2:
            break
