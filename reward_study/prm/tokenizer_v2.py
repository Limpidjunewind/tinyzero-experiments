"""
Number-aware tokenizer for arithmetic expressions (v2).

Key improvement over v1: numbers are tokenized as whole tokens instead of
individual characters. "80 - 35 = 45" becomes [CLS, NUM, -, NUM, =, NUM]
with numeric values [_, 80, _, 35, _, 45] stored separately.

This allows the model to understand that 44 and 45 are close values,
rather than treating them as different character sequences.
"""

import re
from typing import List, Tuple


class NumberAwareTokenizer:
    """Tokenizer that treats numbers as atomic tokens with numeric values."""

    # Token types
    PAD = "[PAD]"
    CLS = "[CLS]"
    UNK = "[UNK]"
    NUM = "[NUM]"   # placeholder for any number

    # Operator tokens
    OPERATORS = ["+", "-", "*", "/", "=", "(", ")"]

    def __init__(self, max_length=32, max_number_value=100000):
        self.max_length = max_length
        self.max_number_value = max_number_value

        # Build vocabulary (small and fixed)
        self.vocab = [self.PAD, self.CLS, self.UNK, self.NUM] + self.OPERATORS
        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for i, t in enumerate(self.vocab)}

        self.pad_id = self.token2id[self.PAD]
        self.cls_id = self.token2id[self.CLS]
        self.unk_id = self.token2id[self.UNK]
        self.num_id = self.token2id[self.NUM]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[Tuple[str, float]]:
        """Split text into (token_type, number_value) pairs.

        For operator tokens, number_value is 0.0.
        For number tokens, number_value is the actual numeric value.

        Example:
            "80 - 35 = 45" -> [("[NUM]", 80), ("-", 0), ("[NUM]", 35), ("=", 0), ("[NUM]", 45)]
        """
        tokens = []
        text = text.strip()

        # Tokenize using regex: numbers (including negative and float) and operators
        pattern = r'(-?\d+\.?\d*)|([+\-*/=()])'
        for match in re.finditer(pattern, text):
            num_str, op_str = match.groups()
            if num_str is not None:
                try:
                    value = float(num_str)
                    tokens.append((self.NUM, value))
                except ValueError:
                    tokens.append((self.UNK, 0.0))
            elif op_str is not None:
                tokens.append((op_str, 0.0))

        return tokens

    def encode(self, text: str, add_cls=True):
        """Encode text to token ids and number values.

        Args:
            text: arithmetic expression (e.g., "80 - 35 = 45")

        Returns:
            token_ids: list of token type ids
            number_values: list of numeric values (normalized to [-1, 1] range)
        """
        raw_tokens = self._tokenize(text)

        token_ids = []
        number_values = []

        if add_cls:
            token_ids.append(self.cls_id)
            number_values.append(0.0)

        for token_type, value in raw_tokens:
            tid = self.token2id.get(token_type, self.unk_id)
            token_ids.append(tid)
            # Normalize number value to reasonable range
            norm_value = value / self.max_number_value if tid == self.num_id else 0.0
            number_values.append(norm_value)

        # Truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            number_values = number_values[:self.max_length]

        return token_ids, number_values

    def encode_padded(self, text: str, add_cls=True):
        """Encode with padding to max_length.

        Returns:
            token_ids: list of length max_length
            number_values: list of length max_length
            attention_mask: list of 1s and 0s
        """
        token_ids, number_values = self.encode(text, add_cls)
        seq_len = len(token_ids)
        attention_mask = [1] * seq_len

        # Pad
        pad_len = self.max_length - seq_len
        token_ids = token_ids + [self.pad_id] * pad_len
        number_values = number_values + [0.0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        return token_ids, number_values, attention_mask
