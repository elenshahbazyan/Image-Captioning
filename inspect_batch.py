# inspect_batch.py
import matplotlib.pyplot as plt
import torch
from preprocess import Vocabulary
from data_loader import get_loader

# load vocab you saved
vocab = Vocabulary.load("preprocessed/vocab.pkl")

# create loader (match what you used)
loader = get_loader(
    images_root="data/coco/train2014/train2014",
    ann_file="data/coco/annotations_trainval2014/annotations/captions_train2014.json",
    vocab=vocab,
    batch_size=8,
    num_workers=0,
    train=False
)

# get one batch
images, captions, lengths = next(iter(loader))

# helper to decode one caption (remove <start> and <end>)
def decode_caption(vocab, cap_tensor, length):
    ids = cap_tensor[:length].tolist()
    # optionally remove start/end tokens if present
    if ids and ids[0] == vocab.word2idx[vocab.START]:
        ids = ids[1:]
    if ids and ids[-1] == vocab.word2idx[vocab.END]:
        ids = ids[:-1]
    words = [vocab.idx2word.get(i, "<unk>") for i in ids]
    return " ".join(words)

# un-normalize for display (ImageNet means/std used in transforms)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# display first 4 images + captions
for i in range(min(4, images.size(0))):
    img = images[i].cpu() * std + mean            # unnormalize
    img = img.permute(1,2,0).numpy()              # CHW -> HWC
    cap_text = decode_caption(vocab, captions[i], lengths[i])
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(cap_text)
    plt.axis('off')
plt.show()