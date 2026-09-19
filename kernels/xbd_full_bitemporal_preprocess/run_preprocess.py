import os
import sys

CODE_CANDIDATES = (
    "/kaggle/input/datasets/amanjolly1994/ieee-access-hurricane-code",
    "/kaggle/input/ieee-access-hurricane-code",
)
XBD_CANDIDATES = (
    "/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd",
    "/kaggle/input/xbd-dataset/xbd",
    "/kaggle/input/datasets/qianlanzz/xbd-dataset",
    "/kaggle/input/xbd-dataset",
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


def _first_existing(candidates, is_xbd_root=False):
    for c in candidates:
        print(f"trying candidate: {c}")
        if os.path.isdir(c) and (not is_xbd_root or os.path.isdir(os.path.join(c, "train"))):
            return c
    return None


_print_input_tree()

code_root = _first_existing(CODE_CANDIDATES)
xbd_root = _first_existing(XBD_CANDIDATES, is_xbd_root=True)

if code_root is None or xbd_root is None:
    raise FileNotFoundError(
        f"code_root={code_root}, xbd_root={xbd_root} -- check the /kaggle/input tree above "
        "and update CODE_CANDIDATES/XBD_CANDIDATES."
    )
print(f"using code_root={code_root}, xbd_root={xbd_root}")

sys.path.insert(0, code_root)
from xbd_preprocess import run  # noqa: E402

n = run(
    xbd_root,
    "/kaggle/working/xbd_full_bitemporal_patches",
    all_disasters=True,
    include_pre_disaster=True,
)
print(f"Experiment F preprocessing done: {n} bitemporal building pairs written.")
