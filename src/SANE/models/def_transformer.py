"""
Code mostly taken / inspired by Andrey Kaparthy's NanoGpt\
https://github.com/karpathy/nanoGPT/blob/master/model.py

His References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        assert d_model % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = d_model
        self.dropout = dropout
        self.causal = causal
        self.block_size = block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # generates a lower triangular matrices. Used to maks out 'future' information from attention matrix
            if self.causal:
                self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(block_size, block_size)).view(
                        1, 1, block_size, block_size
                    ),
                )
        else:
            # all three methods should be enables anyways, but let's make it explicit here
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            )

    def forward(self, x, mask=None):
        (
            B,  # B: batch_size
            T,  # T: token_number
            C,  # C: token_embedding_dim
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # move head forward to be the batch dim
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs) # (nH = self.n_head), hs = C//nh (token_embedding_dim per head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.causal,  # we currently don't want / need autoregressive / causal attention
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    self.bias[:, :, :T, :T] == 0, float("-inf")
                )  # causal attention: masking future tokens, upper right triangular matrix
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.attn = SelfAttention(d_model, n_head, dropout, bias, causal, block_size)
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model=d_model, dropout=dropout, bias=bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        d_model: int = 768,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = False,
        block_size: int = 128,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.transformer = nn.ModuleList(
            [
                Block(d_model, n_head, dropout, bias, causal, block_size)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.transformer:
            x = block(x, mask)
        return x
