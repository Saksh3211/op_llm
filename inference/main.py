"""
Minimal inference entry point: loads the latest checkpoint and lets you
type prompts, printing the model's generated continuation.

This is NOT the full agent loop yet (no tool calls, no plan/check steps) -
it's just enough to confirm the trained model actually generates text.
agent.py + tools/ come later once training quality is worth building on.

Run:
    python inference/main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.model_runtime import InferenceRuntime, find_latest_checkpoint


def main():
    checkpoints_dir = ROOT / "training" / "checkpoints"
    ckpt_path = find_latest_checkpoint(checkpoints_dir)
    runtime = InferenceRuntime(ckpt_path)

    print("\nType a prompt and press Enter. Type 'exit' to quit.\n")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        for piece in runtime.generate_stream(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
            print(piece, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
