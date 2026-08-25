import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
EXCLUDE_DIRS = {"checkpoints", "outputs_debug", "data"}

errors = []

for dirpath, dirnames, filenames in os.walk(ROOT):
    parts = set(dirpath.replace(ROOT, "").split(os.sep))
    if parts & EXCLUDE_DIRS:
        continue
    for fname in filenames:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(dirpath, fname)
        rel = os.path.relpath(fpath, ROOT)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
            ast.parse(src)
        except Exception as e:
            errors.append((rel, "SyntaxError", str(e)))

if errors:
    print("IMPORT CHECK: ERRORS FOUND")
    for rel, typ, msg in errors:
        print(f"- {rel}: {typ}: {msg}")
    sys.exit(2)
else:
    print("IMPORT CHECK: SYNTAX OK (no execution)")
    sys.exit(0)
