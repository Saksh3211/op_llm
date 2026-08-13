"""Simple safe calculator tool.

Provides `evaluate(expr: str) -> str` which evaluates mathematical expressions
in a restricted AST-evaluation environment.
"""

from __future__ import annotations

import ast
import operator as op

# supported operators
_ops = {
	ast.Add: op.add,
	ast.Sub: op.sub,
	ast.Mult: op.mul,
	ast.Div: op.truediv,
	ast.Pow: op.pow,
	ast.USub: op.neg,
	ast.UAdd: op.pos,
	ast.Mod: op.mod,
}


def _eval(node):
	if isinstance(node, ast.Num):
		return node.n
	if isinstance(node, ast.Constant):
		return node.value
	if isinstance(node, ast.BinOp):
		left = _eval(node.left)
		right = _eval(node.right)
		return _ops[type(node.op)](left, right)
	if isinstance(node, ast.UnaryOp):
		operand = _eval(node.operand)
		return _ops[type(node.op)](operand)
	raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def evaluate(expr: str) -> str:
	"""Evaluate a numeric expression safely and return result as string."""
	try:
		node = ast.parse(expr, mode="eval").body
		val = _eval(node)
		return str(val)
	except Exception as e:
		return f"<calc-error: {e}>"
