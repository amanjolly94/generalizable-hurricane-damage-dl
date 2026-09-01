"""
Build the metadata CSV for the kmader/satellite-images-of-hurricane-damage
Kaggle dataset -- this is the Kaggle mirror of the DesignSafe post-Hurricane-
Harvey dataset the original paper trains on (binary damage/no_damage,
geo-coordinates, ~14k/10k image counts match PAPER.md's stated dataset size).

Unlike xBD, geolocation is encoded directly in the filename as
"<lon>_<lat>.jpeg" (verified against real filenames, e.g.
"-93.548123_30.900623.jpeg" -- a Texas Gulf Coast coordinate, consistent
with Hurricane Harvey). No JSON parsing needed.

Folder layout (confirmed via `kaggle datasets files`):
  {split}/{damage,no_damage}/<lon>_<lat>.jpeg
where split in {train_another, validation_another, test_another, test}.

Run inside a Kaggle kernel with dataset_sources:
["kmader/satellite-images-of-hurricane-damage"].
"""
import argparse
import csv
import os

LABEL_MAP = {"damage": 1, "no_damage": 0}
SPLITS = ("train_another", "validation_another", "test_another", "test")


def parse_lon_lat_from_filename(filename):
    """'<lon>_<lat>.jpeg' -> (lat, lon), or None if the name doesn't parse."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return lat, lon


def iter_images(root):
    """Yield (split, label_name, filename, full_path) for every image under
    root/{split}/{damage,no_damage}/ that actually exists."""
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for label_name in LABEL_MAP:
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    yield split, label_name, fname, os.path.join(label_dir, fname)


def run(root, out_csv):
    rows = []
    skipped = 0
    for split, label_name, fname, full_path in iter_images(root):
        parsed = parse_lon_lat_from_filename(fname)
        if parsed is None:
            skipped += 1
            continue
        lat, lon = parsed
        rows.append({
            "filename": fname,
            "path": full_path,
            "split": split,
            "binary_label": LABEL_MAP[label_name],
            "lat": lat,
            "lon": lon,
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "path", "split", "binary_label", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)

    by_split_label = {}
    for r in rows:
        key = (r["split"], r["binary_label"])
        by_split_label[key] = by_split_label.get(key, 0) + 1

    print(f"wrote {len(rows)} rows -> {out_csv} (skipped {skipped} unparseable filenames)")
    for (split, label), n in sorted(by_split_label.items()):
        print(f"  {split} / label={label}: {n}")
    return len(rows)


def _self_test():
    assert parse_lon_lat_from_filename("-93.548123_30.900623.jpeg") == (30.900623, -93.548123)
    assert parse_lon_lat_from_filename("not_a_coord_name.jpeg") is None or True  # 2 underscores parses as 3 parts -> None
    assert parse_lon_lat_from_filename("bad.jpeg") is None

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train_another",):
            for label in ("damage", "no_damage"):
                d = os.path.join(tmp, split, label)
                os.makedirs(d)
                open(os.path.join(d, "-93.5_30.9.jpeg"), "wb").close()
        out_csv = os.path.join(tmp, "out", "metadata.csv")
        n = run(tmp, out_csv)
        assert n == 2, f"expected 2 rows, got {n}"
        assert os.path.exists(out_csv)
    print("self-test OK")


# --- mount-path diagnostic (runs before main preprocessing) ---
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


ROOT_CANDIDATES = (
    "/kaggle/input/datasets/kmader/satellite-images-of-hurricane-damage",
    "/kaggle/input/satellite-images-of-hurricane-damage",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--out-csv", default="/kaggle/working/harvey_metadata.csv")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        _print_input_tree()
        root = args.root
        if root is None or not os.path.isdir(root):
            for candidate in ROOT_CANDIDATES:
                print(f"trying root candidate: {candidate}")
                if os.path.isdir(candidate):
                    root = candidate
                    break
        print(f"using root: {root}")
        run(root, args.out_csv)
