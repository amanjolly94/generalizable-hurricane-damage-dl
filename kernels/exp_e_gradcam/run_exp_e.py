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
MODEL_CANDIDATES = (
    "/kaggle/input/notebooks/amanjolly1994/exp-c-harvey-train/exp_c_harvey/model.keras",
    "/kaggle/input/exp-c-harvey-train/exp_c_harvey/model.keras",
    "/kaggle/input/datasets/amanjolly1994/exp-c-harvey-train/exp_c_harvey/model.keras",
)


def _print_input_tree(path="/kaggle/input", max_depth=6):
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


def _first_existing(candidates, label, check_fn=os.path.isdir):
    for c in candidates:
        print(f"trying {label} candidate: {c}")
        if check_fn(c):
            return c
    return None


_print_input_tree()

code_root = _first_existing(CODE_CANDIDATES, "code")
data_root = _first_existing(DATA_CANDIDATES, "data")
model_path = _first_existing(MODEL_CANDIDATES, "model", check_fn=os.path.exists)

if code_root is None or data_root is None or model_path is None:
    raise FileNotFoundError(
        f"code_root={code_root}, data_root={data_root}, model_path={model_path} -- "
        "check the /kaggle/input tree printed above and update the candidate lists."
    )

print(f"using code_root={code_root}, data_root={data_root}, model_path={model_path}")

sys.path.insert(0, code_root)
from gradcam_eval import run  # noqa: E402

run(
    model_path=model_path,
    data_root=data_root,
    out_dir="/kaggle/working/exp_e_gradcam",
    n_samples=200,
)
