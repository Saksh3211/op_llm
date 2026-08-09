"""Shared schema for processed JSONL shards. Used by prepare_data.py, train.py, evaluate.py."""

from dataclasses import dataclass, asdict
import json


@dataclass
class Record:
    id: str
    stage: int          # curriculum stage this record belongs to (1-5)
    category: str        # english | programming | debugging | algorithms | math | reasoning | docs | tool_use
    text: str            # full text incl. control tokens, e.g. "[PLAN] ... [CODE] ... [CHECK] ... [ANSWER] ..."
    verified: bool        # did this pass compile/test/symbolic verification
    source: str           # traceability back to raw/

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Record":
        return Record(**json.loads(line))


CONTROL_TOKENS = ["[PLAN]", "[CODE]", "[CHECK]", "[ANSWER]"]
CATEGORIES = [
    "english", "programming", "debugging", "algorithms",
    "math", "reasoning", "docs", "tool_use",
]
