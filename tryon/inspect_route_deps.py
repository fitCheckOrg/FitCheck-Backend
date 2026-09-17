"""
Final targeted check before writing api/tryon.py: confirms the exact
import path for get_current_user_id, and looks for an existing
request-body-model convention for simple action endpoints (to decide
garment_id as a JSON body field vs a query param, based on real
precedent rather than assumption).

Run: python -m tryon.inspect_route_deps
"""

import ast
import os

TARGET_FILES = [
    "shared/dependencies.py",
    "api/outfit.py",       # has POST /compatibility, /analyze — simple action endpoints, useful precedent
    "api/social.py",       # has POST /follow/{target_user_id} — simple-ID action endpoint
]


def inspect_file(path):
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    if not os.path.exists(path):
        print(f"  NOT FOUND at {path}")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  Could not parse: {e}")
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0:
            start = node.lineno - 1
            end = node.end_lineno
            print(f"\n{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}(...)")
            print("\n".join(source_lines[start:end]))
            print()


def main():
    for path in TARGET_FILES:
        inspect_file(path)

    print(f"\n>>> Confirm: (1) exact function name/signature of get_current_user_id "
          f"in shared/dependencies.py, (2) whether simple action endpoints with a "
          f"single ID pass it as a query param, path param, or small Pydantic body "
          f"model — use whichever the real precedent shows.")


if __name__ == "__main__":
    main()