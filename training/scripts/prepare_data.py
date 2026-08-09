"""
Turns files under training/data/raw/<category>/ into JSONL shards under
training/data/processed/, following the Record schema.

This is a minimal starting version: one raw file -> one record, wrapped with
control tokens. Extend with real cleaning/filtering/verification per spec
section 5 (compile/run/verify before keeping a record).

Run:
    python training/scripts/prepare_data.py
"""

import sys
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


def wrap_with_control_tokens(text: str, category: str) -> str:
    if category in ("programming", "debugging", "algorithms"):
        return f"[PLAN]\n[CODE]\n{text}\n[CHECK]\n[ANSWER]\n"
    return f"[ANSWER]\n{text}\n"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    buffer: list[Record] = []
    record_id = 0
    total = 0

    def flush():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        out_path = PROCESSED_DIR / f"shard_{shard_idx:04d}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in buffer:
                f.write(rec.to_json() + "\n")
        print(f"wrote {len(buffer)} records -> {out_path}")
        shard_idx += 1
        buffer = []

    for category in CATEGORIES:
        cat_dir = RAW_DIR / category
        if not cat_dir.exists():
            continue
        for file_path in cat_dir.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"skip {file_path}: {e}")
                continue
            if not text:
                continue

            record_id += 1
            rec = Record(
                id=f"{category}_{record_id:06d}",
                stage=DEFAULT_STAGE,
                category=category,
                text=wrap_with_control_tokens(text, category),
                verified=False,   # not yet run through compile/test/symbolic checks
                source=str(file_path.relative_to(ROOT)),
            )
            buffer.append(rec)
            total += 1

            if len(buffer) >= SHARD_SIZE:
                flush()

    flush()
    print(f"Done. {total} records total across {shard_idx} shard(s).")
    if total == 0:
        print(f"No files found. Put source files under {RAW_DIR}/<category>/ first, "
              f"e.g. {RAW_DIR}/programming/example.py")


if __name__ == "__main__":
    main()
