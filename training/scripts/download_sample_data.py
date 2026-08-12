"""
Downloads a small (~10-15 MB) public dataset into training/data/raw/,
sorted into the category folders prepare_data.py already expects.

Sources (all permissively licensed):
  - english:    Tiny Shakespeare (public domain text) - karpathy/char-rnn
  - programming/algorithms: TheAlgorithms/Python (MIT license)
  - math:       GSM8K grade-school math (MIT license) - openai/grade-school-math
  - docs:       CPython tutorial docs (PSF license) - python/cpython

Run:
    python training/scripts/download_sample_data.py

After it finishes:
    python training/scripts/prepare_data.py
    python training/scripts/train.py
"""

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "training" / "data" / "raw"


def download(url: str) -> bytes:
    print(f"downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path} ({len(text.encode('utf-8')):,} bytes)")


def fetch_tinyshakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = download(url).decode("utf-8", errors="ignore")
    save_text(RAW_DIR / "english" / "tinyshakespeare.txt", text)


def fetch_gsm8k():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    raw = download(url).decode("utf-8", errors="ignore")

    out_dir = RAW_DIR / "math"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    lines_out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        question = rec.get("question", "").strip()
        answer = rec.get("answer", "").strip()
        if not question or not answer:
            continue
        lines_out.append(f"Question: {question}\nAnswer: {answer}\n")
        count += 1

    save_text(out_dir / "gsm8k.txt", "\n".join(lines_out))
    print(f"  ({count} Q/A pairs)")


def fetch_algorithms(max_bytes: int = 6_000_000):
    """Downloads TheAlgorithms/Python and keeps only .py files, capped at
    max_bytes total so this stays a 'small, fast iteration' dataset."""
    url = "https://codeload.github.com/TheAlgorithms/Python/zip/refs/heads/master"
    zip_bytes = download(url)

    out_dir = RAW_DIR / "programming"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".py"):
                continue
            if total >= max_bytes:
                break
            data = zf.read(info.filename)
            total += len(data)
            written += 1
            # flatten path into filename to avoid deep nested folders
            flat_name = info.filename.replace("/", "__")
            out_path = out_dir / flat_name
            out_path.write_bytes(data)

    print(f"  wrote {written} .py files, {total:,} bytes total -> {out_dir}")


def fetch_python_docs():
    pages = [
        "introduction", "controlflow", "datastructures",
        "modules", "inputoutput", "errors", "classes",
    ]
    out_dir = RAW_DIR / "docs"
    for page in pages:
        url = f"https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/{page}.rst"
        try:
            text = download(url).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  skip {page}: {e}")
            continue
        save_text(out_dir / f"{page}.rst", text)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Tiny Shakespeare (english) ===")
    fetch_tinyshakespeare()

    print("\n=== GSM8K (math) ===")
    fetch_gsm8k()

    print("\n=== TheAlgorithms/Python (programming) ===")
    fetch_algorithms()

    print("\n=== CPython tutorial (docs) ===")
    fetch_python_docs()

    print("\nDone. Next steps:")
    print("  python training/scripts/prepare_data.py")
    print("  python training/scripts/train.py")


if __name__ == "__main__":
    main()
