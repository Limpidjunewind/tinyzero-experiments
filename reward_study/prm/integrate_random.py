"""
Random PRM scorer for ablation study.

Uses the same architecture as the trained PRM but with random (untrained)
weights, producing ~50% accuracy. This serves as a control to test whether
PRM's contribution comes from its learned signal or just from the act of
injecting any step-level reward (even random).
"""

import re
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from model import ProcessRewardModel
from tokenizer import ArithmeticTokenizer


class PRMScorer:
    """Score intermediate reasoning steps using a RANDOM (untrained) PRM."""

    def __init__(self, checkpoint_path=None, device='cpu', prm_weight=0.1):
        self.device = torch.device(device)
        self.prm_weight = prm_weight
        self.tokenizer = ArithmeticTokenizer(max_length=128)

        # Use same architecture as trained PRM but DO NOT load weights
        self.model = ProcessRewardModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.0,
            pad_id=self.tokenizer.pad_id,
        ).to(self.device)

        # Random initialization only — no checkpoint loading
        self.model.eval()
        print(f"[PRM-RANDOM] Initialized with random weights (NO checkpoint loaded), weight={self.prm_weight}")

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
            steps.append(f"{expr_part} = {result_part}")

        return steps

    @torch.no_grad()
    def score_steps(self, steps):
        """Score a list of arithmetic steps using random PRM."""
        if not steps:
            return []

        batch_ids, batch_masks = [], []
        for step in steps:
            ids, mask = self.tokenizer.encode_padded(step)
            batch_ids.append(ids)
            batch_masks.append(mask)

        ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.long, device=self.device)

        _, probs = self.model(ids_tensor, mask_tensor)
        return probs.squeeze(-1).tolist()

    def compute_process_reward(self, solution_str):
        """Compute the process reward for a full solution string."""
        think_text = self.extract_think_content(solution_str)
        steps = self.extract_steps(think_text)

        if not steps:
            return 0.0, 0, 0.0

        scores = self.score_steps(steps)
        avg_score = sum(scores) / len(scores)

        process_reward = self.prm_weight * (2 * avg_score - 1)

        return process_reward, len(steps), avg_score
