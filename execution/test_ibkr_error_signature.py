#!/usr/bin/env python3
"""Enforcement guard for the IBKR ``EWrapper.error()`` signature gotcha.

Background
----------
IBAPI >= 10.37 calls the ``EWrapper.error`` callback with SIX positional
arguments::

    error(self, reqId, errorTime, errorCode, errorString, advancedOrderReject="")

Any subclass that overrides ``error`` with the older 4-5 argument form crashes
*inside the API reader thread* with ``TypeError: error() takes N positional
arguments but 6 were given``. The visible symptom is usually a misleading
"did not connect / timeout", which is exactly how the cron preflight Check E
silently broke for weeks (see ``MEMORY.md`` -> "IBKR API Gotcha").

Documentation alone failed to prevent recurrence, so this test mechanically
enforces the rule. It scans the repository's Python sources (via AST, no
imports needed) for any ``def error(self, ...)`` whose enclosing class looks
like an EWrapper subclass, and asserts each is the canonical 6-arg signature
OR the tolerant ``(self, reqId, *args)`` form.

Run directly::

    python execution/test_ibkr_error_signature.py

or under pytest::

    python -m pytest execution/test_ibkr_error_signature.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories we intentionally skip:
#   deprecated/ -> retired code, not loaded
#   __pycache__ / caches -> generated
_SKIP_DIR_PARTS = {"deprecated", "__pycache__", ".git", ".ruff_cache",
                   ".pytest_cache", ".ipynb_checkpoints", "node_modules"}

# The two positional args that MUST come first (names checked). The bug we
# guard against is a MISSING ``errorTime`` 2nd positional; the remaining
# arg names (errorCode/errorString vs code/msg) are stylistic and not
# enforced. What matters is arity: 5 required + 1 optional = 6 positional.
_REQUIRED_LEADING = ["self", "reqId", "errorTime"]
_MIN_POSITIONAL = 5  # self, reqId, errorTime, <code>, <msg>  (advancedReject optional)


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in _SKIP_DIR_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def _class_is_ewrapper(node: ast.ClassDef) -> bool:
    """True if the class bases mention EWrapper (covers ``EWrapper`` and
    ``ibapi.wrapper.EWrapper`` style references)."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "EWrapper":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "EWrapper":
            return True
    return False


def _signature_ok(func: ast.FunctionDef) -> bool:
    """Accept the IBAPI >= 10.37 form or the tolerant ``(self, reqId, *args)``
    form. The only hard requirement is the ``errorTime`` 2nd positional plus
    sufficient arity; downstream arg NAMES (errorCode/errorString vs code/msg)
    are not enforced."""
    args = func.args
    pos = [a.arg for a in args.posonlyargs] + [a.arg for a in args.args]
    # Tolerant form: def error(self, reqId, *args)
    if args.vararg is not None and pos[:2] == ["self", "reqId"]:
        return True
    # Canonical form: must start self, reqId, errorTime and have >= 5 positionals.
    if pos[:3] != _REQUIRED_LEADING:
        return False
    return len(pos) >= _MIN_POSITIONAL


def find_bad_error_signatures() -> list[str]:
    """Return a list of human-readable violations (empty == all good)."""
    violations: list[str] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            # Non-importable sketches/snippets are not our concern.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _class_is_ewrapper(node):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "error":
                    if not _signature_ok(item):
                        params = [a.arg for a in item.args.args]
                        rel = path.relative_to(REPO_ROOT)
                        violations.append(
                            f"{rel}:{item.lineno} class {node.name}.error"
                            f"({', '.join(params)}) — expected 6-arg "
                            f"(self, reqId, errorTime, errorCode, errorString, "
                            f"advancedOrderReject='')"
                        )
    return violations


def test_all_ewrapper_error_signatures_are_six_arg() -> None:
    """Every EWrapper subclass must use the IBAPI >= 10.37 error() signature."""
    violations = find_bad_error_signatures()
    assert not violations, (
        "Found EWrapper.error() overrides with the wrong signature. IBAPI "
        ">= 10.37 requires (self, reqId, errorTime, errorCode, errorString, "
        "advancedOrderReject=''). Fix each:\n  - " + "\n  - ".join(violations)
    )


if __name__ == "__main__":
    bad = find_bad_error_signatures()
    if bad:
        print("FAIL — wrong EWrapper.error() signatures found:")
        for v in bad:
            print(f"  - {v}")
        raise SystemExit(1)
    print("PASS — all EWrapper.error() overrides use the 6-arg signature.")
