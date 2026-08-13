"""Very small local search tool over the repository's docs and README.

`search(query: str) -> str` returns a short summary of matching file names
and snippets. This is intentionally lightweight and offline-only (no web).
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def _gather_search_paths() -> list[Path]:
	# prioritize README and training docs and data/raw/docs
	candidates = [ROOT / "README.md", ROOT / "data" / "raw", ROOT / "data"]
	paths: list[Path] = []
	for c in candidates:
		if c.exists():
			if c.is_dir():
				paths.extend(sorted(c.rglob("*.md")))
				paths.extend(sorted(c.rglob("*.rst")))
				paths.extend(sorted(c.rglob("*.txt")))
			else:
				paths.append(c)
	return paths


def _snip_text(text: str, query: str, ctx: int = 100) -> str:
	idx = text.lower().find(query.lower())
	if idx == -1:
		return ""
	start = max(0, idx - ctx)
	end = min(len(text), idx + len(query) + ctx)
	return text[start:end].replace("\n", " ")


def search(query: str, max_results: int = 5) -> str:
	if not query:
		return "<empty-query>"
	paths = _gather_search_paths()
	results = []
	qlow = query.lower()
	for p in paths:
		try:
			text = p.read_text(encoding="utf-8", errors="ignore")
		except Exception:
			continue
		if qlow in text.lower():
			snippet = _snip_text(text, query)
			results.append((p, snippet))
		if len(results) >= max_results:
			break

	if not results:
		return f"<no-local-results for '{query}'>"

	out_lines = []
	for p, sn in results:
		out_lines.append(f"{p.relative_to(ROOT)}: {sn}")
	return "\n".join(out_lines)

