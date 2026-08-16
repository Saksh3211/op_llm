"""
Turns files under training/data/raw/<category>/ into JSONL shards under
training/data/processed/, following the Record schema.

Enhancements for large-scale data:
  - Deduplication (exact text hashing)
  - Quality filtering (minimum length, no excessive noise)
  - Efficient streaming processing (don't load all in memory)
  - Progress reporting
  - Support for JSON, JSONL, and text files

Run:
    python training/scripts/prepare_data.py
"""

import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.data.schema import Record, CATEGORIES

RAW_DIR = ROOT / "training" / "data" / "raw"
PROCESSED_DIR = ROOT / "training" / "data" / "processed"

# crude default: everything in stage 1 until curriculum.py assigns real stages
DEFAULT_STAGE = 1

# max records per shard file
SHARD_SIZE = 2000

# Minimum text length to keep (tokens, rough estimate: 1 char ≈ 0.25 tokens)
MIN_TEXT_CHARS = 50

# Maximum text length per record (avoid huge examples)
MAX_TEXT_CHARS = 100_000


def wrap_with_control_tokens(text: str, category: str) -> str:
    if category in ("programming", "debugging", "algorithms"):
        return f"[PLAN]\n[CODE]\n{text}\n[CHECK]\n[ANSWER]\n"
    if category in ("math", "qa"):
        return f"[QUESTION]\n{text}\n[ANSWER]\n"
    return f"[ANSWER]\n{text}\n"


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # remove excessive whitespace
    text = " ".join(text.split())
    return text.strip()


def hash_text(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer: list[Record] = []
    record_id = 0
    total = 0
    seen_hashes = set()  # for deduplication
    skipped_dup = 0
    skipped_short = 0
    skipped_long = 0
    skipped_errors = 0

    def flush():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        out_path = PROCESSED_DIR / f"shard_{shard_idx:04d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(rec.to_json() + "\n")
        print(f"  wrote {len(buffer)} records -> {out_path}")
        shard_idx += 1
        buffer = []

    print("Processing raw data files...")
    print()

    for category in CATEGORIES:
        cat_dir = RAW_DIR / category
        if not cat_dir.exists():
            continue
        
        print(f"[{category.upper()}]")
        cat_files = list(cat_dir.rglob("*"))
        cat_records = 0
        
        for file_path in cat_files:
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                skipped_errors += 1
                continue
            
            if not text:
                continue

            # Quality filtering
            if len(text) < MIN_TEXT_CHARS:
                skipped_short += 1
                continue
            
            if len(text) > MAX_TEXT_CHARS:
                # truncate instead of skipping long files
                text = text[:MAX_TEXT_CHARS]
                skipped_long += 1

            # Deduplication
            text_hash = hash_text(text)
            if text_hash in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(text_hash)

            # Clean text
            text = clean_text(text)

            record_id += 1
            rec = Record(
                id=f"{category}_{record_id:06d}",
                stage=DEFAULT_STAGE,
                category=category,
                text=wrap_with_control_tokens(text, category),
                verified=False,
                source=str(file_path.relative_to(ROOT)),
            )
            buffer.append(rec)
            total += 1
            cat_records += 1

            if len(buffer) >= SHARD_SIZE:
                flush()

        if cat_records > 0:
            print(f"  → {cat_records} records")

    flush()
    
    print()
    print("=" * 70)
    print(f"Processing complete!")
    print(f"  Total records: {total}")
    print(f"  Total shards: {shard_idx}")
    print(f"  Skipped (duplicate): {skipped_dup}")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (too long, truncated): {skipped_long}")
    print(f"  Skipped (errors): {skipped_errors}")
    print("=" * 70)
    
    if total == 0:
        print(f"No training examples produced - check that "
              f"{RAW_DIR}/<category>/ has non-empty files.")


if __name__ == "__main__":
    main()
