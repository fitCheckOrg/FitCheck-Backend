"""
Targeted inspection of pipelines/avatar_pipeline.py and
pipelines/wardrobe_pipeline.py — specifically their storage/service
boundaries. Answers: who owns fetching the garment, fetching the
processed avatar, saving a rendered result, and generating a public
URL — so tryon_pipeline.py can reuse those boundaries instead of
inventing a parallel storage mechanism.

Narrow by design, same discipline as the previous inspection.

Run: python -m tryon.inspect_pipeline_boundaries
"""

import ast
import os

TARGET_FILES = ["pipelines/avatar_pipeline.py", "pipelines/wardrobe_pipeline.py"]


def find_function_defs(tree, source_lines):
    """Returns every top-level function/async function with its
    full source, plus every call expression inside it (to see what
    external services/helpers it invokes)."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
           node.col_offset == 0:  # top-level only, not nested
            start = node.lineno - 1
            end = node.end_lineno
            snippet = "\n".join(source_lines[start:end])

            calls = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Attribute):
                        calls.add(sub.func.attr)
                    elif isinstance(sub.func, ast.Name):
                        calls.add(sub.func.id)

            results.append({
                "name": node.name,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "source": snippet,
                "calls": sorted(calls),
            })
    return results


def find_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in node.names]
            imports.append((node.module, names))
    return imports


def inspect_file(path):
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    if not os.path.exists(path):
        print(f"  NOT FOUND at {path}")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    print(f"\n--- IMPORTS ---")
    for module, names in find_imports(tree):
        print(f"  from {module} import {', '.join(names)}")

    print(f"\n--- FUNCTIONS ---")
    functions = find_function_defs(tree, source_lines)
    for fn in functions:
        prefix = "async def" if fn["is_async"] else "def"
        print(f"\n{prefix} {fn['name']}(...)")
        print(f"  calls: {', '.join(fn['calls'])}")
        print(f"  --- source ---")
        print(fn["source"])
        print()


def main():
    for path in TARGET_FILES:
        inspect_file(path)

    print(f"\n>>> Look specifically for: which function fetches the processed "
          f"image (avatar photo / garment clean image)? Which function "
          f"uploads/saves a result to storage and returns a URL? Reuse those "
          f"exact functions in tryon_pipeline.py rather than reimplementing "
          f"storage access.")


if __name__ == "__main__":
    main()