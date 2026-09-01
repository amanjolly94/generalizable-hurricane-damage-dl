import os

def walk(path, depth, max_depth=3):
    if depth > max_depth or not os.path.isdir(path):
        return
    entries = sorted(os.listdir(path))[:15]
    for e in entries:
        full = os.path.join(path, e)
        print("  " * depth + e + ("/" if os.path.isdir(full) else ""))
        if os.path.isdir(full):
            walk(full, depth + 1, max_depth)

print("=== /kaggle/input ===")
walk("/kaggle/input", 0)
