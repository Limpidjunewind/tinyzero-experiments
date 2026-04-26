"""
Character-level tokenizer for arithmetic expressions.
Handles digits, operators, parentheses, equals sign, and special tokens.
"""


class ArithmeticTokenizer:
    """Simple character-level tokenizer for arithmetic expressions."""

    # Special tokens
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, max_length=128):
        self.max_length = max_length

        # Build vocabulary: special tokens + digits + operators + misc
        self.vocab = [
            self.PAD_TOKEN,  # 0
            self.CLS_TOKEN,  # 1
            self.UNK_TOKEN,  # 2
        ]
        # Digits
        for d in "0123456789":
            self.vocab.append(d)
        # Operators and symbols
        for s in "+-*/=().":
            self.vocab.append(s)
        # Space and minus sign (for negative numbers)
        self.vocab.append(" ")

        # Build mappings
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}

        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.cls_id = self.token2id[self.CLS_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text, add_cls=True):
        """Encode text to token ids.

        Args:
            text: arithmetic expression string (e.g., "80 - 35 = 45")
            add_cls: whether to prepend [CLS] token

        Returns:
            List of token ids
        """
        ids = []
        if add_cls:
            ids.append(self.cls_id)

        for char in text:
            if char in self.token2id:
                ids.append(self.token2id[char])
            else:
                ids.append(self.unk_id)

        # Truncate if too long
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        return ids

    def encode_padded(self, text, add_cls=True):
        """Encode and pad to max_length.

        Returns:
            token_ids: list of length max_length
            attention_mask: list of 1s and 0s
        """
        ids = self.encode(text, add_cls=add_cls)
        attention_mask = [1] * len(ids)

        # Pad
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        return ids, attention_mask

    def decode(self, ids):
        """Decode token ids back to text."""
        tokens = []
        for idx in ids:
            if idx == self.pad_id:
                break
            if idx == self.cls_id:
                continue
            tokens.append(self.id2token.get(idx, self.UNK_TOKEN))
        return "".join(tokens)
