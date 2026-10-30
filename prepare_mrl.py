import os
import shutil
import sys
from collections import Counter

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join("datasets", "eyes")
folders = {"0": "Close-Eyes", "1": "Open-Eyes"}
for name in folders.values():
    os.makedirs(os.path.join(dst, name), exist_ok=True)

counts, subjects, skipped = Counter(), set(), 0
for root, _, files in os.walk(src):
    for f in files:
        if not f.lower().endswith(".png"):
            continue
        parts = f.split("_")
        if len(parts) < 8 or parts[4] not in folders:
            skipped += 1
            continue
        target = os.path.join(dst, folders[parts[4]], f)
        try:
            os.link(os.path.join(root, f), target)   # fast, no extra disk space
        except OSError:
            shutil.copy2(os.path.join(root, f), target)
        counts[folders[parts[4]]] += 1
        subjects.add(parts[0])

print(f"Images: {dict(counts)}  subjects: {len(subjects)}  skipped: {skipped}")