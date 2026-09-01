import os
import sys

CODE_CANDIDATES = (
    "/kaggle/input/datasets/amanjolly1994/ieee-access-hurricane-code",
    "/kaggle/input/ieee-access-hurricane-code",
)
DATA_CANDIDATES = (
    "/kaggle/input/datasets/kmader/satellite-images-of-hurricane-damage",
    "/kaggle/input/satellite-images-of-hurricane-damage",
)


def _print_input_tree(path="/kaggle/input", max_depth=4):
    def walk(p, depth):
        if depth > max_depth or not os.path.isdir(p):
            return
        for e in sorted(os.listdir(p))[:10]:
            full = os.path.join(p, e)
            print("  " * depth + e + ("/" if os.path.isdir(full) else ""))
            if os.path.isdir(full):
                walk(full, depth + 1)
    print("=== /kaggle/input tree ===")
    walk(path, 0)


def _first_existing(candidates, label):
    for c in candidates:
        print(f"trying {label} candidate: {c}")
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"no {label} candidate found among {candidates}")


_print_input_tree()

code_root = _first_existing(CODE_CANDIDATES, "code")
data_root = _first_existing(DATA_CANDIDATES, "data")
print(f"using code_root={code_root}, data_root={data_root}")

sys.path.insert(0, code_root)
from baseline_train import run  # noqa: E402

run(root=data_root, epochs=8, out_dir="/kaggle/working/exp_c_baselines")
