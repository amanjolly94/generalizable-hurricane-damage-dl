"""
Preprocess the xBD dataset (Kaggle: qianlanzz/xbd-dataset) into per-building
image patches + a metadata CSV, matching the input format of the original
paper's model (per-building satellite crop + (lat, lon) geo coordinate).

Filters to the 4 hurricane events in xBD (Harvey, Florence, Matthew, Michael)
to support Exp A (cross-hurricane generalization) and Exp B (multi-class
severity) from EXPERIMENT_DESIGN.md.

Run inside a Kaggle kernel with dataset_sources: ["qianlanzz/xbd-dataset"].
Local run without that mount will just find zero label files and exit.
"""
import argparse
import csv
import json
import os
import zlib

from PIL import Image
from shapely import wkt as shapely_wkt

HURRICANES = {
    "hurricane-harvey",
    "hurricane-florence",
    "hurricane-matthew",
    "hurricane-michael",
}

# "un-classified" buildings (mostly from pre-disaster labels, or post-disaster
# buildings the annotator couldn't assess) are dropped -- ambiguous ground truth.
SUBTYPE_TO_BINARY = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 1,
    "destroyed": 1,
}
SUBTYPE_TO_SEVERITY = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

TARGET_SIZE = (128, 128)
CROP_PADDING_PX = 4  # small margin around each building's pixel bounding box


def iter_post_disaster_labels(xbd_root):
    """Yield (split, label_path, image_path, pre_image_path) for every
    *_post_disaster.json under xbd_root/{train,test,hold}/labels/ whose
    sibling post-disaster image exists. pre_image_path is set if the
    matching *_pre_disaster.png also exists, else None."""
    for split in ("train", "test", "hold"):
        labels_dir = os.path.join(xbd_root, split, "labels")
        images_dir = os.path.join(xbd_root, split, "images")
        if not os.path.isdir(labels_dir):
            continue
        for fname in os.listdir(labels_dir):
            if not fname.endswith("_post_disaster.json"):
                continue
            label_path = os.path.join(labels_dir, fname)
            image_path = os.path.join(images_dir, fname.replace(".json", ".png"))
            if not os.path.exists(image_path):
                continue
            pre_image_path = image_path.replace("_post_disaster.png", "_pre_disaster.png")
            if not os.path.exists(pre_image_path):
                pre_image_path = None
            yield split, label_path, image_path, pre_image_path


def extract_buildings(label_path):
    """Parse one label JSON, return (disaster, [building_dict, ...]).

    Each building_dict has: uid, subtype, lat, lon, bbox (x0, y0, x1, y1).
    Buildings with an unrecognized/un-classified subtype or a degenerate
    polygon are skipped.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    disaster = data.get("metadata", {}).get("disaster", "")
    features = data.get("features", {})
    xy_by_uid = {f["properties"]["uid"]: f for f in features.get("xy", [])}
    lnglat_by_uid = {f["properties"]["uid"]: f for f in features.get("lng_lat", [])}

    buildings = []
    for uid, xy_feat in xy_by_uid.items():
        subtype = xy_feat["properties"].get("subtype", "")
        if subtype not in SUBTYPE_TO_BINARY:
            continue
        lnglat_feat = lnglat_by_uid.get(uid)
        if lnglat_feat is None:
            continue

        xy_poly = shapely_wkt.loads(xy_feat["wkt"])
        lnglat_poly = shapely_wkt.loads(lnglat_feat["wkt"])
        if xy_poly.is_empty or lnglat_poly.is_empty:
            continue

        minx, miny, maxx, maxy = xy_poly.bounds
        centroid = lnglat_poly.centroid

        buildings.append({
            "uid": uid,
            "subtype": subtype,
            "lat": centroid.y,
            "lon": centroid.x,
            "bbox": (minx, miny, maxx, maxy),
        })
    return disaster, buildings


def crop_building_patch(image, bbox, padding=CROP_PADDING_PX, target_size=TARGET_SIZE):
    """Crop+resize one building's patch from a PIL image, clamped to image bounds."""
    w, h = image.size
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - padding)
    y0 = max(0, int(y0) - padding)
    x1 = min(w, int(x1) + padding)
    y1 = min(h, int(y1) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1)).resize(target_size, Image.BILINEAR)


def run(xbd_root, out_dir, limit=None, all_disasters=False, include_pre_disaster=False,
        shard_index=None, shard_count=None):
    """all_disasters=True processes every xView2 disaster type instead of just
    the 4 hurricanes (Experiment F: full-xView2 comparison against Zarski &
    Miszczak 2024). include_pre_disaster=True also crops and saves the
    pre-disaster image for the same building bbox, needed for the bitemporal
    model variant (build_full_model_bitemporal in model.py) -- the building
    footprint is identical pre/post, so the post-disaster label's bbox is
    reused for the pre-disaster crop.

    shard_index/shard_count split the label files across shard_count parallel
    Kaggle kernels (across different accounts) by a stable hash of each
    label filename -- not Python's built-in hash(), which is randomized per
    process (PYTHONHASHSEED) and would give a different, inconsistent split
    on every run. Every label file is assigned to exactly one shard, so
    running all shard_count shards and merging their outputs reproduces the
    unsharded result."""
    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    rows = []
    n_written = 0
    fieldnames = [
        "patch_filename", "source_image", "disaster", "split", "uid",
        "lat", "lon", "binary_label", "severity_label", "subtype",
    ]
    if include_pre_disaster:
        fieldnames.append("pre_patch_filename")

    for split, label_path, image_path, pre_image_path in iter_post_disaster_labels(xbd_root):
        if shard_count:
            fname = os.path.basename(label_path)
            if zlib.crc32(fname.encode("utf-8")) % shard_count != shard_index:
                continue
        disaster, buildings = extract_buildings(label_path)
        if (not all_disasters and disaster not in HURRICANES) or not buildings:
            continue
        if include_pre_disaster and pre_image_path is None:
            continue  # can't build a bitemporal pair without the pre-disaster image

        image = Image.open(image_path).convert("RGB")
        pre_image = Image.open(pre_image_path).convert("RGB") if include_pre_disaster else None
        source_id = os.path.basename(image_path).replace("_post_disaster.png", "")

        for b in buildings:
            patch = crop_building_patch(image, b["bbox"])
            if patch is None:
                continue
            pre_patch_filename = None
            if include_pre_disaster:
                pre_patch = crop_building_patch(pre_image, b["bbox"])
                if pre_patch is None:
                    continue
                pre_patch_filename = f"{source_id}_{b['uid']}_pre.png"
                pre_patch.save(os.path.join(patches_dir, pre_patch_filename))

            patch_filename = f"{source_id}_{b['uid']}.png"
            patch.save(os.path.join(patches_dir, patch_filename))
            row = {
                "patch_filename": patch_filename,
                "source_image": source_id,
                "disaster": disaster,
                "split": split,
                "uid": b["uid"],
                "lat": b["lat"],
                "lon": b["lon"],
                "binary_label": SUBTYPE_TO_BINARY[b["subtype"]],
                "severity_label": SUBTYPE_TO_SEVERITY[b["subtype"]],
                "subtype": b["subtype"],
            }
            if include_pre_disaster:
                row["pre_patch_filename"] = pre_patch_filename
            rows.append(row)
            n_written += 1
            if limit and n_written >= limit:
                break
        if limit and n_written >= limit:
            break

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {n_written} building patches -> {patches_dir}")
    print(f"wrote metadata -> {csv_path}")
    return n_written


def _self_test():
    """No xBD data needed -- verifies the WKT/centroid/bbox/crop math in isolation."""
    import tempfile

    square_xy = "POLYGON ((10 10, 10 30, 30 30, 30 10, 10 10))"
    square_lnglat = "POLYGON ((-79.10 33.50, -79.10 33.51, -79.09 33.51, -79.09 33.50, -79.10 33.50))"

    xy_poly = shapely_wkt.loads(square_xy)
    lnglat_poly = shapely_wkt.loads(square_lnglat)
    minx, miny, maxx, maxy = xy_poly.bounds
    assert (minx, miny, maxx, maxy) == (10, 10, 30, 30), f"bbox wrong: {xy_poly.bounds}"

    centroid = lnglat_poly.centroid
    assert abs(centroid.x - (-79.095)) < 1e-6, f"centroid lon wrong: {centroid.x}"
    assert abs(centroid.y - 33.505) < 1e-6, f"centroid lat wrong: {centroid.y}"

    assert SUBTYPE_TO_BINARY["no-damage"] == 0
    assert SUBTYPE_TO_BINARY["destroyed"] == 1
    assert SUBTYPE_TO_SEVERITY["major-damage"] == 2
    assert "un-classified" not in SUBTYPE_TO_BINARY

    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    patch = crop_building_patch(img, (10, 10, 30, 30), padding=2, target_size=(16, 16))
    assert patch is not None and patch.size == (16, 16), "crop/resize failed"

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "xbd", "train", "labels"))
        os.makedirs(os.path.join(tmp, "xbd", "train", "images"))
        label = {
            "metadata": {"disaster": "hurricane-harvey", "img_name": "x_00000000_post_disaster.png"},
            "features": {
                "xy": [{"properties": {"feature_type": "building", "subtype": "destroyed", "uid": "u1"}, "wkt": square_xy}],
                "lng_lat": [{"properties": {"feature_type": "building", "subtype": "destroyed", "uid": "u1"}, "wkt": square_lnglat}],
            },
        }
        label_path = os.path.join(tmp, "xbd", "train", "labels", "x_00000000_post_disaster.json")
        with open(label_path, "w") as f:
            json.dump(label, f)
        Image.new("RGB", (64, 64)).save(os.path.join(tmp, "xbd", "train", "images", "x_00000000_post_disaster.png"))

        out_dir = os.path.join(tmp, "out")
        n = run(os.path.join(tmp, "xbd"), out_dir)
        assert n == 1, f"expected 1 patch, got {n}"
        assert os.path.exists(os.path.join(out_dir, "metadata.csv"))

        # all_disasters=True must still find the same building even though
        # "hurricane-harvey" would also pass the default HURRICANES filter --
        # use a non-hurricane disaster to prove the filter is actually bypassed.
        label["metadata"]["disaster"] = "socal-fire"
        with open(label_path, "w") as f:
            json.dump(label, f)
        out_dir_all = os.path.join(tmp, "out_all")
        n_default = run(os.path.join(tmp, "xbd"), out_dir_all + "_default")
        assert n_default == 0, f"non-hurricane disaster should be skipped by default, got {n_default}"
        n_all = run(os.path.join(tmp, "xbd"), out_dir_all, all_disasters=True)
        assert n_all == 1, f"all_disasters=True should still process non-hurricane disasters, got {n_all}"

        # include_pre_disaster=True needs a sibling *_pre_disaster.png to produce a pair.
        out_dir_pre = os.path.join(tmp, "out_pre")
        n_no_pre_image = run(os.path.join(tmp, "xbd"), out_dir_pre + "_missing", all_disasters=True, include_pre_disaster=True)
        assert n_no_pre_image == 0, f"missing pre-disaster image should skip the building, got {n_no_pre_image}"
        Image.new("RGB", (64, 64)).save(os.path.join(tmp, "xbd", "train", "images", "x_00000000_pre_disaster.png"))
        n_with_pre = run(os.path.join(tmp, "xbd"), out_dir_pre, all_disasters=True, include_pre_disaster=True)
        assert n_with_pre == 1, f"expected 1 bitemporal pair, got {n_with_pre}"
        with open(os.path.join(out_dir_pre, "metadata.csv"), newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
            assert row["pre_patch_filename"].endswith("_pre.png"), f"pre_patch_filename missing/wrong: {row}"
            assert os.path.exists(os.path.join(out_dir_pre, "patches", row["pre_patch_filename"]))

    # Sharding: running every shard of shard_count and merging must reproduce
    # the unsharded result exactly, with no building double-counted or dropped.
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "xbd", "train", "labels"))
        os.makedirs(os.path.join(tmp, "xbd", "train", "images"))
        n_source_images = 9
        for i in range(n_source_images):
            uid = f"u{i}"
            label = {
                "metadata": {"disaster": "socal-fire", "img_name": f"x_{i:08d}_post_disaster.png"},
                "features": {
                    "xy": [{"properties": {"feature_type": "building", "subtype": "no-damage", "uid": uid}, "wkt": square_xy}],
                    "lng_lat": [{"properties": {"feature_type": "building", "subtype": "no-damage", "uid": uid}, "wkt": square_lnglat}],
                },
            }
            with open(os.path.join(tmp, "xbd", "train", "labels", f"x_{i:08d}_post_disaster.json"), "w") as f:
                json.dump(label, f)
            Image.new("RGB", (64, 64)).save(os.path.join(tmp, "xbd", "train", "images", f"x_{i:08d}_post_disaster.png"))

        unsharded_dir = os.path.join(tmp, "unsharded")
        n_unsharded = run(os.path.join(tmp, "xbd"), unsharded_dir, all_disasters=True)
        assert n_unsharded == n_source_images, f"expected {n_source_images} buildings, got {n_unsharded}"

        shard_count = 3
        total_sharded = 0
        seen_uids = set()
        for shard_index in range(shard_count):
            shard_dir = os.path.join(tmp, f"shard_{shard_index}")
            n_shard = run(os.path.join(tmp, "xbd"), shard_dir, all_disasters=True,
                          shard_index=shard_index, shard_count=shard_count)
            total_sharded += n_shard
            with open(os.path.join(shard_dir, "metadata.csv"), newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    assert row["uid"] not in seen_uids, f"uid {row['uid']} assigned to more than one shard"
                    seen_uids.add(row["uid"])
        assert total_sharded == n_unsharded, (
            f"sharded total ({total_sharded}) != unsharded total ({n_unsharded}) -- shards must partition, not sample"
        )

    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xbd-root", default="/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd")
    parser.add_argument("--out-dir", default="/kaggle/working/xbd_hurricane_patches")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--all-disasters", action="store_true",
                         help="Process all 17 xView2 disaster types instead of just the 4 hurricanes (Experiment F).")
    parser.add_argument("--include-pre-disaster", action="store_true",
                         help="Also crop and save the pre-disaster image per building, for the bitemporal model variant (Experiment F).")
    parser.add_argument("--shard-index", type=int, default=None,
                         help="This kernel's shard number (0-based), for splitting the job across multiple parallel Kaggle accounts.")
    parser.add_argument("--shard-count", type=int, default=None,
                         help="Total number of shards; every label file goes to exactly one of them, deterministically.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        run(args.xbd_root, args.out_dir, limit=args.limit,
            all_disasters=args.all_disasters, include_pre_disaster=args.include_pre_disaster,
            shard_index=args.shard_index, shard_count=args.shard_count)
