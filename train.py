import os
import argparse
import csv
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from preprocess import Vocabulary, get_transforms
from data_loader import get_loader
from feature_extractor import EncoderCNN
from decoder import DecoderRNN
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_plot(batch_steps, batch_losses, epoch_losses, out_path="preprocessed/training_plot.png"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(10,5))

    # top: batch-step losses (no smoothing)
    plt.subplot(1,2,1)
    if batch_steps:
        plt.plot(batch_steps, batch_losses)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Batch loss (every log_step)")
    else:
        plt.text(0.5, 0.5, "No batch losses recorded", ha='center', va='center')

    # right: epoch average losses
    plt.subplot(1,2,2)
    if epoch_losses:
        epochs = list(range(1, len(epoch_losses)+1))
        plt.plot(epochs, epoch_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Avg loss")
        plt.title("Epoch average loss")
    else:
        plt.text(0.5, 0.5, "No epoch losses recorded", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved training plot to {out_path}")

def save_csv(batch_steps, batch_losses, epoch_losses, out_csv="preprocessed/training_loss.csv"):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "epoch"])
        # write batch entries (epoch unknown here for each entry, we leave epoch blank)
        for step, loss in zip(batch_steps, batch_losses):
            writer.writerow([step, loss, ""])
        # write epoch averages at the end
        writer.writerow([])
        writer.writerow(["epoch", "avg_loss"])
        for i, val in enumerate(epoch_losses, 1):
            writer.writerow([i, val])
    print(f"Saved training losses to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", type=str, default="data/coco/train2014/train2014")
    parser.add_argument("--ann_file", type=str, default="data/coco/annotations_trainval2014/annotations/captions_train2014.json")
    parser.add_argument("--vocab_path", type=str, default="preprocessed/vocab.pkl")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=0)  # use 0 on Windows
    parser.add_argument("--train_cnn", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocab
    vocab = Vocabulary.load(args.vocab_path)
    print("Vocab size:", len(vocab))

    # Data loader
    transform = get_transforms(train=True)
    train_loader = get_loader(
        images_root=args.images_root,
        ann_file=args.ann_file,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        train=True
    )

    # Models
    encoder = EncoderCNN(embed_size=args.embed_size, train_cnn=args.train_cnn).to(device)
    decoder_dropout = 0.0 if args.num_layers == 1 else 0.3
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), num_layers=args.num_layers, dropout=decoder_dropout).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    total_steps = len(train_loader)
    print(f"Starting training: {args.num_epochs} epochs, {total_steps} steps/epoch")

    # For plotting / tracking
    batch_steps = []     # global "step index" for plotting
    batch_losses = []    # loss recorded at each log_step
    epoch_losses = []    # average loss per epoch

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        it = enumerate(train_loader)
        if args.use_tqdm:
            it = tqdm(it, total=total_steps, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for i, (images, captions, lengths) in it:
            global_step += 1
            images = images.to(device)
            captions = captions.to(device)
            # targets = captions[:,1:], target lengths = lengths - 1
            try:
                targets_packed = pack_padded_sequence(captions[:, 1:], lengths - 1, batch_first=True, enforce_sorted=False)[0]
            except Exception:
                lengths_cpu = lengths.cpu().tolist()
                targets_packed = pack_padded_sequence(captions[:, 1:], torch.tensor([l - 1 for l in lengths_cpu]), batch_first=True, enforce_sorted=False)[0]

            # Forward
            features = encoder(images)
            outputs = decoder(features, captions)

            try:
                outputs_packed = pack_padded_sequence(outputs[:, 1:, :], lengths - 1, batch_first=True, enforce_sorted=False)[0]
            except Exception:
                lengths_cpu = lengths.cpu().tolist()
                outputs_packed = pack_padded_sequence(outputs[:, 1:, :], torch.tensor([l - 1 for l in lengths_cpu]), batch_first=True, enforce_sorted=False)[0]

            loss = criterion(outputs_packed, targets_packed)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            # record and optionally print
            if global_step % args.log_step == 0:
                batch_steps.append(global_step)
                batch_losses.append(loss.item())
                if not args.use_tqdm:
                    print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{total_steps}], GlobalStep [{global_step}], Loss: {loss.item():.4f}")

            if args.use_tqdm:
                it.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, total_steps)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_path": args.vocab_path,
            "args": vars(args)
        }
        ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(ckpt, best_path)
            print(f"Saved best model: {best_path} (loss {best_loss:.4f})")

    # After training: save plot and CSV
    plot_path = "preprocessed/training_plot.png"
    csv_path = "preprocessed/training_loss.csv"
    save_plot(batch_steps, batch_losses, epoch_losses, out_path=plot_path)
    save_csv(batch_steps, batch_losses, epoch_losses, out_csv=csv_path)

    print("Training complete ✅")

if __name__ == "__main__":
    main()
