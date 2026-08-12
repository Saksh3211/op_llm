"""
Minimal inference entry point: loads the latest checkpoint and lets you
type prompts, printing the model's generated continuation as it streams.

This is NOT the full agent loop yet (no tool calls, no plan/check steps) -
it's just enough to confirm the trained model actually generates text.
agent.py + tools/ come later once training quality is worth building on.

Commands (type these instead of a prompt):
    /temp             - show current temperature mode
    /temp 0.7         - lock temperature to a fixed value (for testing)
    /temp auto        - go back to adaptive (task-based) temperature
    exit              - quit

Run:
    python inference/main.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.model_runtime import InferenceRuntime, find_latest_checkpoint
from inference.temperature import TemperatureController


def handle_temp_command(arg: str, controller: TemperatureController):
    arg = arg.strip().lower()
    if not arg:
        if controller.override is not None:
            print(f"Temperature is fixed at {controller.override}")
        else:
            print("Temperature is adaptive (auto, based on prompt type)")
        return
    if arg == "auto":
        controller.set_override(None)
        print("Temperature set to adaptive (auto)")
        return
    try:
        value = float(arg)
        if not (0.0 < value <= 2.0):
            print("Please use a value between 0 and 2, e.g. /temp 0.7")
            return
        controller.set_override(value)
        print(f"Temperature fixed at {value}")
    except ValueError:
        print("Usage: /temp 0.7   or   /temp auto   or   /temp")


def main():
    checkpoints_dir = ROOT / "training" / "checkpoints"
    ckpt_path = find_latest_checkpoint(checkpoints_dir)
    runtime = InferenceRuntime(ckpt_path)
    temp_controller = TemperatureController()

    print("\nType a prompt and press Enter. Type 'exit' to quit.")
    print("Type '/temp 0.7' to fix temperature, '/temp auto' to go back to adaptive, '/temp' to check current mode.\n")

    while True:
        prompt = input("> ").strip()
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue
        if prompt.startswith("/temp"):
            handle_temp_command(prompt[len("/temp"):], temp_controller)
            continue

        temperature, source = temp_controller.get(prompt)
        print(f"[temperature={temperature} ({source})]")

        for piece in runtime.generate_stream(prompt, max_new_tokens=200, temperature=temperature, top_k=40):
            print(piece, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
