"""
Step 1 of API integration. Inspects the existing FastAPI backend's
route/service conventions — response shape, error handling pattern,
status codes actually in use — so the new try-on endpoint follows
the project's existing style rather than introducing a second one.

Pure inspection. No new routes, no schema decisions yet.

Run: python -m tryon.inspect_api_conventions
"""

import os
import re
import glob

# Adjust if the backend lives elsewhere relative to this file
BACKEND_SEARCH_PATHS = [
    "main.py",
    "**/*.py",
]
EXCLUDE_DIRS = {"venv", "tryon", "tests", "__pycache__", ".git", "outputs"}


def find_python_files():
    files = []
    for root, dirs, filenames in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                files.append(os.path.join(root, f))
    return files


def scan_file(path):
    findings = {
        "routes": [],
        "response_patterns": [],
        "error_patterns": [],
        "status_codes": [],
    }
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return findings

    # Route decorators
    for m in re.finditer(r'@\w+\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', content):
        findings["routes"].append((m.group(1).upper(), m.group(2), path))

    # Common response shape hints: {"success": ..., "data": ...} pattern 
    # already confirmed in use (seen throughout this session's API calls)
    for m in re.finditer(r'\{["\']success["\']:\s*\w+.{0,80}', content):
        findings["response_patterns"].append((m.group(0)[:100], path))

    # HTTPException usage
    for m in re.finditer(r'HTTPException\([^)]{0,150}\)', content):
        findings["error_patterns"].append((m.group(0)[:150], path))

    # status_code= occurrences
    for m in re.finditer(r'status_code\s*=\s*(\d{3})', content):
        findings["status_codes"].append((m.group(1), path))

    return findings


def main():
    print("Scanning for FastAPI conventions in the backend...\n")

    all_files = find_python_files()
    print(f"Scanning {len(all_files)} Python files (excluding {EXCLUDE_DIRS})\n")

    all_routes = []
    all_response_patterns = []
    all_error_patterns = []
    all_status_codes = []

    for path in all_files:
        findings = scan_file(path)
        all_routes.extend(findings["routes"])
        all_response_patterns.extend(findings["response_patterns"])
        all_error_patterns.extend(findings["error_patterns"])
        all_status_codes.extend(findings["status_codes"])

    print(f"=== EXISTING ROUTES ({len(all_routes)} found) ===")
    for method, route, path in all_routes:
        print(f"  {method:6} {route:<40} ({path})")

    print(f"\n=== RESPONSE SHAPE PATTERNS ({len(all_response_patterns)} found) ===")
    seen = set()
    for pattern, path in all_response_patterns[:15]:
        key = pattern[:40]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {pattern}  ({path})")

    print(f"\n=== ERROR HANDLING PATTERNS ({len(all_error_patterns)} found) ===")
    seen = set()
    for pattern, path in all_error_patterns[:15]:
        key = pattern[:40]
        if key in seen:
            continue
        seen.add(key)
        print(f"  {pattern}  ({path})")

    print(f"\n=== STATUS CODES IN USE ===")
    from collections import Counter
    code_counts = Counter(code for code, _ in all_status_codes)
    for code, count in sorted(code_counts.items()):
        print(f"  {code}: used {count}x")

    print(f"\n>>> Use these findings to match the try-on endpoint's request/response/error "
          f"shape to what's ALREADY established, rather than inventing a new convention.")


if __name__ == "__main__":
    main()