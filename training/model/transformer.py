"""Assembled decoder-only transformer with weight tying."""

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock, precompute_rope


class SmallLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        head_dim = cfg.dim // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_embeddings(tokens)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.output(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
