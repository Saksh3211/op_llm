"""
Main training loop.

On start, asks how many hours you want to train for, then stops itself
(after finishing the current step + saving a checkpoint) once that time
is used up - you don't have to babysit it or guess step counts.

Run:
    python training/scripts/train.py
"""

import sys
import time
import glob
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# TPU support (optional, for Google Colab and TPU-equipped systems)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.model.config import ModelConfig
from training.model.transformer import SmallLM
from training.data.schema import Record


PAD_ID = 0  # token id used for padding short sequences in a batch


# ---------------------------------------------------------------------------
# Time budget prompt
# ---------------------------------------------------------------------------

def ask_time_budget_hours() -> float:
    """Ask the user how long to train for. Accepts e.g. '2', '2.5', '0.25'."""
    while True:
        raw = input("How many hours should this training run last? ").strip()
        try:
            hours = float(raw)
            if hours <= 0:
                print("Enter a number greater than 0.")
                continue
            return hours
        except ValueError:
            print("Please enter a number, e.g. 2 or 1.5")


# ---------------------------------------------------------------------------
# Placeholder tokenizer - swap this out once tokenizer_train.py exists.
# Character/byte-level so the pipeline runs end-to-end with zero setup.
# Byte 0 is reserved as PAD, so real bytes are shifted up by 1.
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        # +1 so raw byte 0 doesn't collide with PAD_ID; clip to vocab range
        return [min(b + 1, self.vocab_size - 1) for b in text.encode("utf-8", errors="ignore")]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShardDataset(Dataset):
    """Reads all JSONL shards under training/data/processed/, tokenizes, and
    yields variable-length (up to seq_len) token blocks for next-token
    prediction. Short records are kept as-is (not dropped) and padded per
    batch by the collate function below - this matters for small test
    datasets where most records are shorter than seq_len."""

    def __init__(self, processed_dir: Path, tokenizer: CharTokenizer, seq_len: int, stage: int | None = None):
        self.seq_len = seq_len
        self.examples: list[list[int]] = []

        shard_files = sorted(glob.glob(str(processed_dir / "*.jsonl")))
        if not shard_files:
            raise FileNotFoundError(
                f"No .jsonl shards found in {processed_dir}. "
                "Run prepare_data.py first."
            )

        for path in shard_files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = Record.from_json(line)
                    if stage is not None and rec.stage != stage:
                        continue
                    ids = tokenizer.encode(rec.text)
                    if len(ids) < 2:
                        continue  # need at least 1 input + 1 target token

                    # split into chunks of up to seq_len+1 tokens; the tail
                    # chunk (shorter than seq_len) is kept, not dropped
                    step = seq_len  # non-overlapping chunks
                    for i in range(0, len(ids), step):
                        chunk = ids[i:i + seq_len + 1]
                        if len(chunk) >= 2:
                            self.examples.append(chunk)

        if not self.examples:
            raise ValueError(
                "No training examples produced - check that "
                f"{processed_dir} has non-empty records."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def collate_pad(batch):
    """Pads x with PAD_ID and y with -100 (ignored by cross_entropy) so
    variable-length examples can share a batch tensor."""
    max_len = max(x.size(0) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad_amt = max_len - x.size(0)
        if pad_amt > 0:
            x = torch.cat([x, torch.full((pad_amt,), PAD_ID, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_amt,), -100, dtype=torch.long)])
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# Device selection - TPU, CUDA, DirectML, or CPU (in priority order)
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    # Priority order: TPU > CUDA > DirectML > CPU
    
    # Try TPU first (Google Colab, TPU-equipped systems)
    if HAS_TPU:
        try:
            device = xm.xla_device()
            print(f"Using TPU device: {device}")
            print("  TPU is excellent for LLM training - 2-3x faster than GPU!")
            return device
        except RuntimeError:
            pass  # TPU not available, try next option
    
    # Try CUDA GPU (NVIDIA)
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    
    # Try DirectML (AMD GPU on Windows)
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
    
    # Fallback to CPU
    print("Using CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    hours = ask_time_budget_hours()
    time_budget_seconds = hours * 3600
    print(f"Training for up to {hours} hour(s) ({time_budget_seconds:.0f}s).")

    processed_dir = ROOT / "training" / "data" / "processed"
    checkpoints_dir = ROOT / "training" / "checkpoints"
    logs_dir = ROOT / "training" / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    tokenizer = CharTokenizer(vocab_size=cfg.vocab_size)

    dataset = ShardDataset(processed_dir, tokenizer, seq_len=cfg.max_seq_len)
    print(f"Loaded {len(dataset)} training examples.")

    batch_size = min(32, len(dataset))  # don't ask for a bigger batch than we have data
    loader = DataLoader(dataset, batch_size=min(16, batch_size), shuffle=True, drop_last=False, collate_fn=collate_pad)

    device = pick_device()
    model = SmallLM(cfg).to(device)
    print(f"Model parameters: {model.num_params():,}")
    
    # TPU initialization: synchronize processes if using TPU
    if HAS_TPU and "xla" in str(device):
        xm.rendezvous("init")  # synchronize across TPU cores

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    log_path = logs_dir / f"train_{int(time.time())}.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    checkpoint_every_seconds = 10 * 60   # save every 10 min, in addition to on-exit
    last_checkpoint_time = time.time()

    start_time = time.time()
    step = 0
    tokens_seen = 0
    elapsed = 0.0

    def save_checkpoint(tag: str):
        path = checkpoints_dir / f"ckpt_{tag}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "step": step,
            "tokens_seen": tokens_seen,
        }, path)
        print(f"Saved checkpoint: {path}")

    print("Starting training. Press Ctrl+C to stop early (checkpoint will still be saved).")
    stopped_reason = "completed"
    try:
        model.train()
        while True:
            for x, y in loader:
                elapsed = time.time() - start_time
                if elapsed >= time_budget_seconds:
                    stopped_reason = "time budget reached"
                    raise StopIteration

                x, y = x.to(device), y.to(device)

                step_start = time.time()
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step_time = time.time() - step_start

                step += 1
                batch_tokens = x.numel()
                tokens_seen += batch_tokens
                toks_per_sec = batch_tokens / max(step_time, 1e-6)
                
                # TPU-specific: synchronize metrics across cores
                if HAS_TPU and "xla" in str(device):
                    xm.rendezvous("step_end")

                if step % 10 == 0 or step == 1:
                    remaining = time_budget_seconds - elapsed
                    print(
                        f"step {step:6d} | loss {loss.item():.4f} | "
                        f"{toks_per_sec:8.1f} tok/s | "
                        f"elapsed {elapsed/60:6.1f}m | remaining {remaining/60:6.1f}m"
                    )
                    log_file.write(json.dumps({
                        "step": step,
                        "loss": loss.item(),
                        "tokens_per_sec": toks_per_sec,
                        "elapsed_s": elapsed,
                        "tokens_seen": tokens_seen,
                    }) + "\n")
                    log_file.flush()

                if time.time() - last_checkpoint_time >= checkpoint_every_seconds:
                    # On TPU, sync to ensure all cores finish before checkpointing
                    if HAS_TPU and "xla" in str(device):
                        xm.mark_step()
                    save_checkpoint("latest")
                    last_checkpoint_time = time.time()

    except StopIteration:
        print(f"\nStopping training ({stopped_reason}).")
    except KeyboardInterrupt:
        print("\nStopping training (interrupted).")
    finally:
        save_checkpoint("final")
        log_file.close()
        total_elapsed = time.time() - start_time
        print(f"Done. Trained {step} steps, {tokens_seen:,} tokens, in {total_elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
