"""
Loads a trained checkpoint and runs autoregressive generation.

Same CharTokenizer as train.py (byte-level, +1 shift, 0 reserved for PAD).
Once tokenizer_train.py exists this should be swapped for a shared,
properly trained tokenizer instead of duplicating this class.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.model.config import ModelConfig
from training.model.transformer import SmallLM

PAD_ID = 0


class CharTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(b + 1, self.vocab_size - 1) for b in text.encode("utf-8", errors="ignore")]

    def decode(self, ids: list[int]) -> str:
        byte_vals = bytes(max(i - 1, 0) for i in ids if i != PAD_ID)
        return byte_vals.decode("utf-8", errors="ignore")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    try:
        import torch_directml
        print("Using DirectML (AMD GPU)")
        return torch_directml.device()
    except ImportError as e:
        print(f"DirectML not available ({e}). "
            f"torch-directml only supports Python 3.8-3.12 - "
            f"check your interpreter version with 'python --version'. "
            f"Falling back to CPU.")
    except Exception as e:
        print(f"DirectML import failed ({e}). Falling back to CPU.")
    print("Using CPU")
    return torch.device("cpu")


class InferenceRuntime:
    def __init__(self, checkpoint_path: Path, device: torch.device | None = None):
        self.device = device or pick_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        cfg = ModelConfig(**ckpt["config"])
        self.cfg = cfg
        self.model = SmallLM(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  trained for {ckpt.get('step', '?')} steps, "
            f"{ckpt.get('tokens_seen', '?')} tokens seen")
        print(f"  device: {self.device}")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 200,
                temperature: float = 0.8, top_k: int | None = 40) -> str:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [1]  # avoid empty sequence

        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            # keep only the last max_seq_len tokens as context
            context = tokens[:, -self.cfg.max_seq_len:]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        generated_ids = tokens[0].tolist()
        return self.tokenizer.decode(generated_ids)

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int = 200,
                        temperature: float = 0.8, top_k: int | None = 40):
        """Same as generate(), but yields (new_text_piece, token_id) tuples
        as each token is produced so callers can show streaming token-level
        output (and token ids) as the model generates."""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [1]

        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        prev_text = ""

        for _ in range(max_new_tokens):
            context = tokens[:, -self.cfg.max_seq_len:]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = v[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            # token id as python int
            token_id = int(next_token[0, 0].item())

            full_text = self.tokenizer.decode(tokens[0].tolist())
            if len(full_text) > len(prev_text):
                new_piece = full_text[len(prev_text):]
                prev_text = full_text
                yield new_piece, token_id


def find_latest_checkpoint(checkpoints_dir: Path) -> Path:
    candidates = sorted(checkpoints_dir.glob("ckpt_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir}. Run training/scripts/train.py first."
        )
    # prefer ckpt_final.pt if present, else most recently modified
    for c in candidates:
        if c.name == "ckpt_final.pt":
            return c
    return max(candidates, key=lambda p: p.stat().st_mtime)
