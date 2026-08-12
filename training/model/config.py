"""Architecture hyperparameters. Tune these based on benchmarking (see spec section 3)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 192             # hidden size
    n_layers: int = 6
    n_heads: int = 6           # query heads
    n_kv_heads: int = 3      # GQA: fewer KV heads than query heads
    ffn_hidden_mult: float = 8 / 3   # SwiGLU convention (~2.67x dim, rounded to multiple below)
    ffn_multiple_of: int = 32
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True   # weight tying: input/output embedding share weights
    dropout: float = 0.0

    def ffn_hidden_dim(self) -> int:
        hidden = int(self.ffn_hidden_mult * self.dim)
        # round up to nearest multiple of ffn_multiple_of
        return self.ffn_multiple_of * ((hidden + self.ffn_multiple_of - 1) // self.ffn_multiple_of)
