"""
Training script for PRM v2 (number-aware model).

Usage:
    python train_v2.py \
        --train_data prm/data/train_v2.jsonl \
        --test_data prm/data/test_v2.jsonl \
        --output_dir prm/checkpoints_v2 \
        --epochs 15 \
        --batch_size 512 \
        --lr 3e-4
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_v2 import ProcessRewardModelV2
from tokenizer_v2 import NumberAwareTokenizer


class ArithmeticDatasetV2(Dataset):
    """Dataset using number-aware tokenizer."""

    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        token_ids, number_values, attention_mask = self.tokenizer.encode_padded(sample['text'])
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'number_values': torch.tensor(number_values, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float),
        }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            number_values = batch['number_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, probs = model(token_ids, number_values, attention_mask)
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = (probs.squeeze(-1) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = NumberAwareTokenizer(max_length=args.max_length)
    print(f"Vocab size: {tokenizer.vocab_size} (number-aware)")

    train_dataset = ArithmeticDatasetV2(args.train_data, tokenizer)
    test_dataset = ArithmeticDatasetV2(args.test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model = ProcessRewardModelV2(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 75)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            token_ids = batch['token_ids'].to(device)
            number_values = batch['number_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, probs = model(token_ids, number_values, attention_mask)
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(labels)
            preds = (probs.squeeze(-1) > 0.5).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += len(labels)

            if (step + 1) % args.log_interval == 0:
                acc = epoch_correct / epoch_samples
                avg_loss = epoch_loss / epoch_samples
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | LR: {lr:.2e}")

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        elapsed = time.time() - t0

        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | {elapsed:.1f}s | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'args': vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> Saved best (acc={best_acc:.4f})")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'args': vars(args),
    }, os.path.join(args.output_dir, "final_model.pt"))

    print("-" * 75)
    print(f"Done. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/fangda/tinyzero/prm/checkpoints_v2")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    train(args)
