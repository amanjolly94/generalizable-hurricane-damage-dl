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
    """Yield (split, label_path, image_path) for every *_post_disaster.json
    under xbd_root/{train,test,hold}/labels/ whose sibling image exists."""
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
            if os.path.exists(image_path):
                yield split, label_path, image_path


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


def run(xbd_root, out_dir, limit=None):
    patches_dir = os.path.join(out_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    rows = []
    n_written = 0

    for split, label_path, image_path in iter_post_disaster_labels(xbd_root):
        disaster, buildings = extract_buildings(label_path)
        if disaster not in HURRICANES or not buildings:
            continue

        image = Image.open(image_path).convert("RGB")
        source_id = os.path.basename(image_path).replace("_post_disaster.png", "")

        for b in buildings:
            patch = crop_building_patch(image, b["bbox"])
            if patch is None:
                continue
            patch_filename = f"{source_id}_{b['uid']}.png"
            patch.save(os.path.join(patches_dir, patch_filename))
            rows.append({
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
            })
            n_written += 1
            if limit and n_written >= limit:
                break
        if limit and n_written >= limit:
            break

    csv_path = os.path.join(out_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_filename", "source_image", "disaster", "split", "uid",
            "lat", "lon", "binary_label", "severity_label", "subtype",
        ])
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

    print("self-test OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xbd-root", default="/kaggle/input/datasets/qianlanzz/xbd-dataset/xbd")
    parser.add_argument("--out-dir", default="/kaggle/working/xbd_hurricane_patches")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        run(args.xbd_root, args.out_dir, limit=args.limit)
