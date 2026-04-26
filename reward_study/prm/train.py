"""
Training script for Process Reward Model.

Usage:
    python train.py \
        --train_data prm/data/train.jsonl \
        --test_data prm/data/test.jsonl \
        --output_dir prm/checkpoints \
        --epochs 10 \
        --batch_size 256 \
        --lr 3e-4
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ProcessRewardModel
from tokenizer import ArithmeticTokenizer


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic step verification."""

    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
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
        text = sample['text']
        label = sample['label']

        token_ids, attention_mask = self.tokenizer.encode_padded(text)

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
        }


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, probs = model(input_ids, attention_mask)
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            predictions = (probs.squeeze(-1) > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train(args):
    """Main training loop."""

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = ArithmeticTokenizer(max_length=args.max_length)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Dataset
    train_dataset = ArithmeticDataset(args.train_data, tokenizer, args.max_length)
    test_dataset = ArithmeticDataset(args.test_data, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = ProcessRewardModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout,
        pad_id=tokenizer.pad_id,
    ).to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Training
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_samples = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits, probs = model(input_ids, attention_mask)
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Stats
            epoch_loss += loss.item() * len(labels)
            predictions = (probs.squeeze(-1) > 0.5).float()
            epoch_correct += (predictions == labels).sum().item()
            epoch_samples += len(labels)

            # Print progress
            if (step + 1) % args.log_interval == 0:
                step_acc = epoch_correct / epoch_samples
                step_loss = epoch_loss / epoch_samples
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Step {step+1}/{len(train_loader)} | "
                      f"Loss: {step_loss:.4f} | "
                      f"Acc: {step_acc:.4f} | "
                      f"LR: {lr:.2e}")

        # Epoch stats
        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        elapsed = time.time() - start_time

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'args': vars(args),
            }, save_path)
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    # Save final model
    save_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'args': vars(args),
    }, save_path)

    print("-" * 70)
    print(f"Training complete. Best test accuracy: {best_acc:.4f}")
    print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Process Reward Model")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/fangda/tinyzero/prm/checkpoints")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    train(args)
