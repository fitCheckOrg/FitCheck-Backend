"""
Targeted inspection of wardrobe.py and avatar.py — the two most
comparable existing domains to try-on. Reads actual source directly
and follows local imports via the ast module (reliable parsing, not
regex), rather than pattern-matching literal text that can silently
miss real conventions (as the previous broader scan did).

Prints: route signatures, response_model declarations, actual
Pydantic model definitions for anything imported locally, and every
HTTPException raise with full context.

Run: python -m tryon.inspect_api_contract
"""

import ast
import os

TARGET_FILES = ["api/wardrobe.py", "api/avatar.py"]


def parse_imports(tree, source_dir):
    """Returns list of (module_path, is_local) for all imports,
    resolving local (project-relative) imports to actual file paths
    where possible."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            # Heuristic: treat anything not starting with a known 
            # stdlib/third-party prefix as potentially local
            candidate_path = module.replace(".", os.sep) + ".py"
            full_path = os.path.join(source_dir, candidate_path)
            # Also try relative to project root
            root_path = candidate_path
            if os.path.exists(full_path):
                imports.append((module, full_path))
            elif os.path.exists(root_path):
                imports.append((module, root_path))
    return imports


def find_route_functions(tree, source_lines):
    """Finds every function decorated with @router.<method>(...),
    returns (method, route, function_source_snippet)."""
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    method = dec.func.attr
                    if method in ("get", "post", "put", "delete", "patch"):
                        route_arg = dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else "?"
                        response_model = None
                        for kw in dec.keywords:
                            if kw.arg == "response_model":
                                response_model = ast.unparse(kw.value)
                        start = node.lineno - 1
                        end = node.end_lineno
                        snippet = "\n".join(source_lines[start:end])
                        routes.append({
                            "method": method.upper(),
                            "route": route_arg,
                            "function_name": node.name,
                            "response_model": response_model,
                            "source": snippet,
                        })
    return routes


def find_http_exceptions(tree, source_lines):
    """Finds every HTTPException(...) call with surrounding context."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "HTTPException":
            start = max(0, node.lineno - 2)
            end = min(len(source_lines), node.end_lineno + 1)
            snippet = "\n".join(source_lines[start:end])
            results.append(snippet)
    return results


def find_class_definitions(tree, source_lines):
    """Finds Pydantic model / class definitions (BaseModel subclasses
    or any class, for inspection)."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno
            snippet = "\n".join(source_lines[start:end])
            results.append((node.name, snippet))
    return results


def inspect_file(path):
    print(f"\n{'='*70}\n{path}\n{'='*70}")
    if not os.path.exists(path):
        print(f"  NOT FOUND at {path}")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    print(f"\n--- ROUTE FUNCTIONS ---")
    routes = find_route_functions(tree, source_lines)
    for r in routes:
        print(f"\n{r['method']} {r['route']}  (function: {r['function_name']})")
        print(f"  response_model: {r['response_model']}")
        print(f"  --- source ---")
        print(f"{r['source']}\n")

    print(f"\n--- HTTPException RAISES ---")
    exceptions = find_http_exceptions(tree, source_lines)
    for e in exceptions:
        print(f"  {e}\n")

    print(f"\n--- IMPORTS (potential response/error models) ---")
    imports = parse_imports(tree, os.path.dirname(path) or ".")
    for module, resolved_path in imports:
        print(f"  {module}  ->  {resolved_path}")

    return imports


def main():
    all_imports = []
    for path in TARGET_FILES:
        imports = inspect_file(path)
        if imports:
            all_imports.extend(imports)

    print(f"\n\n{'='*70}\nFOLLOWING IMPORTS TO SHARED MODELS\n{'='*70}")
    seen_paths = set()
    for module, resolved_path in all_imports:
        if resolved_path in seen_paths or not os.path.exists(resolved_path):
            continue
        seen_paths.add(resolved_path)
        print(f"\n--- {resolved_path} (imported as '{module}') ---")
        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        source_lines = source.splitlines()
        try:
            tree = ast.parse(source)
            classes = find_class_definitions(tree, source_lines)
            for name, snippet in classes:
                print(f"\nclass {name}:")
                print(snippet)
        except SyntaxError:
            print("  (could not parse — skipping)")

    print(f"\n>>> Use the actual route signatures, response_model types, and "
          f"HTTPException patterns above to derive the try-on contract from "
          f"real precedent, not assumption.")


if __name__ == "__main__":
    main()