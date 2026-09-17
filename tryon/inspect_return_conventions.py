"""
Final targeted inspection before writing tryon_pipeline.py. Answers
two specific, narrow questions:

1. What does the image-bytes-to-upload conversion actually look like
   in existing pipelines? (PIL Image -> bytes, what format/params)
2. What's the exact shape of storage.upload()'s return and how is
   it consumed downstream (e.g. what gets put in a response model)?

Run: python -m tryon.inspect_return_conventions
"""

import ast
import os

TARGET_FILES = [
    "core/storage/s3.py",
    "core/image/enhancer.py",
    "shared/models/avatar.py",
    "workers/avatar/saver.py",
    "workers/wardrobe/saver.py",
]


def find_function_defs(tree, source_lines):
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0:
            start = node.lineno - 1
            end = node.end_lineno
            snippet = "\n".join(source_lines[start:end])
            results.append({"name": node.name, "source": snippet})
    return results


def find_class_defs(tree, source_lines):
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno
            snippet = "\n".join(source_lines[start:end])
            results.append({"name": node.name, "source": snippet})
    return results


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

    print(f"\n--- CLASSES ---")
    for c in find_class_defs(tree, source_lines):
        print(f"\nclass {c['name']}:")
        print(c["source"])

    print(f"\n--- FUNCTIONS ---")
    for fn in find_function_defs(tree, source_lines):
        print(f"\ndef/async def {fn['name']}(...)")
        print(fn["source"])


def main():
    for path in TARGET_FILES:
        inspect_file(path)

    print(f"\n>>> Look for: (1) how enhancer.py / other code converts a PIL "
          f"Image or numpy array into raw bytes before calling storage.upload "
          f"— exact method (BytesIO + .save(format=...)?), and (2) how "
          f"saver.py functions consume storage.upload()'s return dict, to "
          f"match TryOnResult's shape to the same convention.")


if __name__ == "__main__":
    main()