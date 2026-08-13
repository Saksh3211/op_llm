"""Tool to read a file path within the repository.

`read_path(path: str) -> str` returns the first N lines of the file or an error.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read_path(p: str, max_lines: int = 200) -> str:
	if not p:
		return "<no-path>"
	candidate = (ROOT / p).resolve()
	try:
		# ensure it's inside repo
		if ROOT not in candidate.parents and candidate != ROOT:
			return "<path-outside-repo>"
		text = candidate.read_text(encoding="utf-8", errors="ignore")
		lines = text.splitlines()
		snippet = "\n".join(lines[:max_lines])
		if len(lines) > max_lines:
			snippet += "\n... (truncated)"
		return snippet
	except FileNotFoundError:
		return f"<file-not-found: {p}>"
	except Exception as e:
		return f"<read-error: {e}>"

