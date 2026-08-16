"""Agent loop for inference using lightweight tools.

Behavior:
 - Parses optional "/think <0-1>" prefix on prompts to control planning depth.
 - Runs a short planning pass (uses model runtime to create a plan text).
 - Extracts simple tool calls from the plan (calculator, search_docs, read_file).
 - Executes tools, appends observations, and asks the model to generate final answer.

This is intentionally lightweight and heuristic-driven so it works offline
without external web APIs. It can be extended later with richer parsers
or a proper tool-invocation format.
"""

from __future__ import annotations

import re
from typing import Optional

from inference.model_runtime import InferenceRuntime

from inference.tools import calculator, search_docs, read_file


class Agent:
	def __init__(self, runtime: InferenceRuntime):
		self.runtime = runtime

	def _parse_think(self, prompt: str) -> tuple[Optional[float], str]:
		# supports: "/think" or "/think 0.7" prefix, optionally with following text
		m = re.match(r"^/think(?:\s+([0-1](?:\.\d+)?))?(?:\s+(.*))?$", prompt.strip())
		if not m:
			return None, prompt
		val = m.group(1)
		rest = m.group(2) or ""
		return (float(val) if val is not None else None), rest

	def _plan(self, prompt: str, think: Optional[float]) -> str:
		# produce a short plan using the model; temperature scaled by think
		temp = max(0.2, min(1.5, think if think is not None else 0.6))
		plan_prompt = (
			"You are an assistant that outlines a short plan of steps to answer the user's"
			" request. Provide 2-6 short bullet steps. If a step requires using a tool,"
			" mention the tool name (calculator, search_docs, read_file) and the query in"
			" simple plain text.\n\nUser prompt:\n" + prompt + "\n\nPlan:\n"
		)
		return self.runtime.generate(plan_prompt, max_new_tokens=120, temperature=temp)

	def _extract_actions(self, plan_text: str) -> list[tuple[str, str]]:
        # split plan into tasks and extract tool calls from user prompt if plan is empty
        if not plan_text or not plan_text.strip():
            # fallback: try to extract from the original user prompt via self
            # but we need to be passed the original prompt. For now, return empty.
            return []

        actions: list[tuple[str, str]] = []

        # split on "and" / "also" / "," to handle multi-task prompts better
        text = plan_text
        # split on 'and' or 'also' or commas, but preserve the text
        chunks = re.split(r"(?:^|\s+)(?:and|also|,)(?:\s+|$)", text, flags=re.I)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # look for explicit tool mentions
            # calculator
            m = re.search(r"calculator[:\-\s]+(.+)$", chunk, flags=re.I)
            if m:
                actions.append(("calculator", m.group(1).strip().strip('"')))
                continue
            # search_docs
            m = re.search(r"search[_ ]?docs[:\-\s]+\"?(.+?)\"?$", chunk, flags=re.I)
            if m:
                actions.append(("search_docs", m.group(1).strip()))
                continue
            # read_file
            m = re.search(r"read[_ ]?file[:\-\s]+(.+)$", chunk, flags=re.I)
            if m:
                actions.append(("read_file", m.group(1).strip().strip('"')))
                continue

            # looser patterns
            m = re.search(r"calculate\s+(.+)$", chunk, flags=re.I)
            if m:
                actions.append(("calculator", m.group(1).strip()))
                continue
            m = re.search(r"search(?: for)?\s+\"?(.+?)\"?$", chunk, flags=re.I)
            if m:
                actions.append(("search_docs", m.group(1).strip()))
                continue
            m = re.search(r"read(?: the)? file\s+\"?(.+?)\"?$", chunk, flags=re.I)
            if m:
                actions.append(("read_file", m.group(1).strip()))
                continue
		return actions

	def _run_tool(self, name: str, arg: str) -> str:
		try:
			if name == "calculator":
				return calculator.evaluate(arg)
			if name == "search_docs":
				return search_docs.search(arg)
			if name == "read_file":
				return read_file.read_path(arg)
		except Exception as e:
			return f"<tool-error {name}: {e}>"
		return f"<unknown-tool {name}>"

	def answer(self, prompt: str, think: Optional[float] = None, max_new_tokens: int = 300):
		# 1) Plan
		plan_text = self._plan(prompt, think)

		# 2) Extract actions
		actions = self._extract_actions(plan_text)
        # fallback: if plan extraction found nothing, try the user prompt directly
        if not actions:
            actions = self._extract_actions(prompt)
		# 3) Build final prompt: include plan and tool observations
		obs_text = "\n".join(f"[{n}] {a} -> {o}" for n, a, o in observations)
		final_prompt = (
			"User: " + prompt + "\n\nPlan:\n" + plan_text + "\n\nObservations:\n" + obs_text
			+ "\n\nFinal Answer:\n"
		)

		# 4) Generate final answer
		out = self.runtime.generate(final_prompt, max_new_tokens=max_new_tokens, temperature=0.7)
		return out, plan_text, observations

	def answer_stream(self, prompt: str, think: Optional[float] = None,
					  max_new_tokens: int = 300, temperature: float = 0.7):
		"""Return (plan_text, observations, generator) where generator yields
		(new_piece, token_id) tuples from the model as it generates the final answer."""
		plan_text = self._plan(prompt, think)
		actions = self._extract_actions(plan_text)
        # fallback: if plan extraction found nothing, try the user prompt directly
        if not actions:
            actions = self._extract_actions(prompt)
		final_prompt = (
			"User: " + prompt + "\n\nPlan:\n" + plan_text + "\n\nObservations:\n" + obs_text
			+ "\n\nFinal Answer:\n"
		)

		gen = self.runtime.generate_stream(final_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
		return plan_text, observations, gen

	def interactive(self):
		print("Agent interactive mode. Prefix prompts with '/think <0-1>' optionally.")
		while True:
			raw = input("> ").strip()
			if raw.lower() in ("exit", "quit"):
				break
			think_val, new_prompt = self._parse_think(raw)
			# if user typed only /think or /think <val> then ask for actual prompt
			if not new_prompt:
				new_prompt = input("Prompt: ").strip()
			print("Planning...")
			answer, plan, observations = self.answer(new_prompt, think=think_val)
			print("\n--- Plan ---\n" + plan)
			if observations:
				print("\n--- Observations ---")
				for n, a, o in observations:
					print(f"[{n}] {a} -> {o}")
			print("\n--- Answer ---\n" + answer + "\n")


def create_agent_from_runtime(runtime:InferenceRuntime) -> Agent:
	return Agent(runtime)


