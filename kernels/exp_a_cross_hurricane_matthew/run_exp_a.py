import os
import sys

CODE_CANDIDATES = (
    "/kaggle/input/datasets/amanjolly1994/ieee-access-hurricane-code",
    "/kaggle/input/ieee-access-hurricane-code",
)
DATA_CANDIDATES = (
    "/kaggle/input/notebooks/amanjolly1994/xbd-hurricane-preprocess/xbd_hurricane_patches",
    "/kaggle/input/xbd-hurricane-preprocess/xbd_hurricane_patches",
    "/kaggle/input/datasets/amanjolly1994/xbd-hurricane-preprocess/xbd_hurricane_patches",
    "/kaggle/input/xbd-hurricane-preprocess",
)


def _print_input_tree(path="/kaggle/input", max_depth=5):
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
        if os.path.isdir(c) and (label != "data" or os.path.exists(os.path.join(c, "metadata.csv"))):
            return c
    return None


_print_input_tree()

code_root = _first_existing(CODE_CANDIDATES, "code")
data_root = _first_existing(DATA_CANDIDATES, "data")

if code_root is None or data_root is None:
    raise FileNotFoundError(
        f"code_root={code_root}, data_root={data_root} -- check the /kaggle/input tree "
        "printed above and update DATA_CANDIDATES/CODE_CANDIDATES."
    )

print(f"using code_root={code_root}, data_root={data_root}")

sys.path.insert(0, code_root)
from train_cross_hurricane import run  # noqa: E402

run(
    root=data_root,
    held_out="hurricane-matthew",
    epochs=30,
    finetune_epochs=10,
    finetune_fraction=0.2,
    out_dir="/kaggle/working/exp_a_cross_hurricane",
)
