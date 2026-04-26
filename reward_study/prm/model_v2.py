"""
Number-aware Transformer for Process Reward Model (v2).

Key improvements over v1:
1. Number value embedding: numbers are not just token IDs but carry their
   actual numeric value, encoded via a small MLP. This means 44 and 45
   have similar embeddings (they are numerically close).
2. Larger capacity: d_model=256, n_layers=6, n_heads=8
3. Pre-LayerNorm architecture for stable training

All core Transformer components remain hand-implemented.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumberValueEncoder(nn.Module):
    """Encode a scalar number value into a d_model-dimensional vector.

    Uses a small MLP to project the normalized number value.
    This ensures that numerically close values (44 vs 45) get similar embeddings,
    unlike character-level tokenization where they would be completely different.
    """

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, values):
        """
        Args:
            values: (batch_size, seq_len) normalized number values

        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        # values: (batch, seq_len) -> (batch, seq_len, 1)
        x = values.unsqueeze(-1)
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise FFN with GELU activation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class ProcessRewardModelV2(nn.Module):
    """Number-aware Process Reward Model.

    Architecture:
        Input → Token Type Embedding + Number Value Embedding + Position Embedding
              → N × TransformerBlock
              → LayerNorm → [CLS] → Classifier

    The key innovation is dual embedding:
    - Token type embedding: what kind of token (number, +, -, *, /, =, etc.)
    - Number value embedding: for NUM tokens, encodes the actual numeric value
      via a small MLP, so 44 and 45 get similar representations.
    """

    def __init__(
        self,
        vocab_size=11,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=32,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Token type embedding (small vocab: PAD, CLS, UNK, NUM, +, -, *, /, =, (, ))
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Number value encoder: scalar -> d_model vector
        self.number_encoder = NumberValueEncoder(d_model, hidden_dim=64)

        # Learnable positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final norm and classifier
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, token_ids, number_values, attention_mask=None):
        """
        Args:
            token_ids: (batch, seq_len) token type indices
            number_values: (batch, seq_len) normalized numeric values
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, 1)
            probs: (batch, 1)
        """
        batch_size, seq_len = token_ids.shape

        if attention_mask is None:
            attention_mask = (token_ids != self.pad_id).long()

        # Token type embedding
        token_emb = self.token_embedding(token_ids)  # (batch, seq, d_model)

        # Number value embedding (adds numeric understanding)
        num_emb = self.number_encoder(number_values)  # (batch, seq, d_model)

        # Position embedding
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq, d_model)

        # Combine: token type + number value + position
        x = token_emb + num_emb + pos_emb
        x = self.embedding_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final norm
        x = self.final_norm(x)

        # [CLS] token classification
        cls_hidden = x[:, 0, :]
        logits = self.classifier(cls_hidden)
        probs = torch.sigmoid(logits)

        return logits, probs

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
