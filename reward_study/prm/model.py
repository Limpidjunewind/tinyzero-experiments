"""
Hand-built Transformer for Process Reward Model (PRM).

A small encoder-only Transformer that classifies whether an arithmetic
expression claim (e.g., "80 - 35 = 45") is correct or not.

All core components are implemented from scratch:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- Transformer Block (Pre-LayerNorm)
- Learnable Positional Encoding
- Full model with classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Computes scaled dot-product attention with multiple heads in parallel.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len), 1 for valid, 0 for padding

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply attention mask (mask padding positions)
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        # context: (batch, n_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Two linear transformations with a GELU activation in between.
    FFN(x) = Linear(GELU(Linear(x)))
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with Pre-LayerNorm architecture.

    Pre-LN is more stable for training than Post-LN:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = self.dropout1(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class ProcessRewardModel(nn.Module):
    """Process Reward Model for arithmetic step verification.

    Architecture:
        Input tokens → Token Embedding + Positional Embedding
        → N × TransformerBlock
        → Final LayerNorm
        → [CLS] hidden state → Linear → Sigmoid → correctness probability

    Args:
        vocab_size: size of the character-level vocabulary
        d_model: hidden dimension (default: 128)
        n_heads: number of attention heads (default: 4)
        n_layers: number of transformer blocks (default: 4)
        d_ff: feed-forward intermediate dimension (default: 512)
        max_seq_len: maximum input sequence length (default: 128)
        dropout: dropout rate (default: 0.1)
        pad_id: padding token id (default: 0)
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learnable positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head: [CLS] hidden state → binary prediction
        self.classifier = nn.Linear(d_model, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for linear layers."""
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

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len) token indices
            attention_mask: (batch_size, seq_len) 1 for valid, 0 for padding

        Returns:
            logits: (batch_size, 1) raw logit for binary classification
            probs: (batch_size, 1) probability of the step being correct
        """
        batch_size, seq_len = input_ids.shape

        # Generate attention mask from padding if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        # Token + Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.final_norm(x)

        # Take [CLS] token (position 0) representation
        cls_hidden = x[:, 0, :]  # (batch_size, d_model)

        # Classification
        logits = self.classifier(cls_hidden)  # (batch_size, 1)
        probs = torch.sigmoid(logits)

        return logits, probs

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
