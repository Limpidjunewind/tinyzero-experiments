"""
PRM integration utilities for PPO training.

Provides functions to:
1. Load the trained PRM model
2. Extract intermediate reasoning steps from model outputs
3. Score steps and produce process reward signals
"""

import re
import torch
import os
import sys

# Ensure prm modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from model import ProcessRewardModel
from tokenizer import ArithmeticTokenizer


class PRMScorer:
    """Score intermediate reasoning steps using the trained PRM."""

    def __init__(self, checkpoint_path, device='cpu', prm_weight=0.1):
        self.device = torch.device(device)
        self.prm_weight = prm_weight

        self.tokenizer = ArithmeticTokenizer(max_length=128)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_args = ckpt.get('args', {})

        self.model = ProcessRewardModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_args.get('d_model', 256),
            n_heads=model_args.get('n_heads', 8),
            n_layers=model_args.get('n_layers', 6),
            d_ff=model_args.get('d_ff', 1024),
            max_seq_len=model_args.get('max_length', 128),
            dropout=0.0,
            pad_id=self.tokenizer.pad_id,
        ).to(self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    @staticmethod
    def extract_think_content(text):
        """Extract content within <think> tags."""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'<think>(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_steps(think_text):
        """Extract arithmetic claim steps from think content."""
        if think_text is None:
            return []

        step_pattern = r'([\d\s+\-*/().]+?)\s*=\s*(-?\d+\.?\d*)'
        matches = re.finditer(step_pattern, think_text)

        steps = []
        for m in matches:
            expr_part = m.group(1).strip()
            result_part = m.group(2).strip()

            if re.match(r'^\d+$', expr_part):
                continue
            if not re.search(r'[+\-*/]', expr_part):
                continue

            step_str = f"{expr_part} = {result_part}"
            steps.append(step_str)

        return steps

    @torch.no_grad()
    def score_steps(self, steps):
        """Score a list of arithmetic steps using PRM."""
        if not steps:
            return []

        batch_ids = []
        batch_masks = []
        for step in steps:
            ids, mask = self.tokenizer.encode_padded(step)
            batch_ids.append(ids)
            batch_masks.append(mask)

        ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.long, device=self.device)

        _, probs = self.model(ids_tensor, mask_tensor)
        return probs.squeeze(-1).tolist()

    def compute_process_reward(self, solution_str):
        """Compute the process reward for a full solution string.

        Returns:
            process_reward: float, scaled process reward
            num_steps: int, number of steps found
            avg_score: float, average PRM score (before scaling)
        """
        think_text = self.extract_think_content(solution_str)
        steps = self.extract_steps(think_text)

        if not steps:
            return 0.0, 0, 0.0

        scores = self.score_steps(steps)
        avg_score = sum(scores) / len(scores)

        # Map [0, 1] -> [-weight, +weight]
        process_reward = self.prm_weight * (2 * avg_score - 1)

        return process_reward, len(steps), avg_score
