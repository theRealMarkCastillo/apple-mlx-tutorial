"""A small decoder for lesson 07b: explicit RoPE, GQA, and cache offsets."""

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, dims, heads, kv_heads, rope):
        super().__init__()
        if dims % heads or heads % kv_heads or (dims // heads) % 2:
            raise ValueError("Use even head dimensions and heads divisible by kv_heads")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dims // heads
        self.q = nn.Linear(dims, dims, bias=False)
        self.k = nn.Linear(dims, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dims, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False) if rope else None

    def __call__(self, x, cache=None):
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache[0].shape[2]
        if self.rope is not None:
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = (
                mx.concatenate([cache[0], k], axis=2),
                mx.concatenate([cache[1], v], axis=2),
            )
        # Query i can see the old prefix and new tokens through offset+i.
        allowed = mx.arange(k.shape[2])[None, :] <= (offset + mx.arange(t))[:, None]
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=allowed
        )
        y = y.transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.out(y), (k, v)


class FeedForward(nn.Module):
    def __init__(self, dims, swiglu):
        super().__init__()
        self.swiglu = swiglu
        hidden = ((8 * dims // 3 + 15) // 16) * 16 if swiglu else 4 * dims
        self.up = nn.Linear(dims, hidden, bias=False)
        self.down = nn.Linear(hidden, dims, bias=False)
        if swiglu:
            self.gate = nn.Linear(dims, hidden, bias=False)

    def __call__(self, x):
        h = nn.silu(self.gate(x)) * self.up(x) if self.swiglu else nn.gelu(self.up(x))
        return self.down(h)


class Block(nn.Module):
    def __init__(self, dims, heads, kv_heads, rope, rmsnorm, swiglu):
        super().__init__()
        norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.norm1, self.norm2 = norm(dims), norm(dims)
        self.attn = Attention(dims, heads, kv_heads, rope)
        self.ffn = FeedForward(dims, swiglu)

    def __call__(self, x, cache):
        y, cache = self.attn(self.norm1(x), cache)
        x = x + y
        return x + self.ffn(self.norm2(x)), cache


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dims=64,
        layers=2,
        heads=4,
        kv_heads=2,
        rope=True,
        rmsnorm=True,
        swiglu=True,
        max_length=512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dims)
        self.positions = None if rope else nn.Embedding(max_length, dims)
        self.blocks = [
            Block(dims, heads, kv_heads, rope, rmsnorm, swiglu) for _ in range(layers)
        ]
        self.norm = (nn.RMSNorm if rmsnorm else nn.LayerNorm)(dims)
        self.head = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, tokens, cache=None, return_cache=False):
        states = [None] * len(self.blocks) if cache is None else cache
        if len(states) != len(self.blocks):
            raise ValueError("Need one cache entry per layer")
        offset = 0 if states[0] is None else states[0][0].shape[2]
        x = self.embed(tokens)
        if self.positions is not None:
            if offset + tokens.shape[1] > self.positions.weight.shape[0]:
                raise ValueError("Learned position table exhausted")
            x = x + self.positions(mx.arange(offset, offset + tokens.shape[1]))
        updated = []
        for block, state in zip(self.blocks, states, strict=True):
            x, state = block(x, state)
            updated.append(state)
        logits = self.head(self.norm(x))
        return (logits, updated) if return_cache else logits


def cache_bytes(cache):
    return sum(array.nbytes for pair in cache for array in pair)
