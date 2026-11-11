"""Export the volumes for the NiiVue web viewer (docs/index.html).

Run from the repo root after `python all_subjects.py` (which saves every subject's transform):
    conda activate mri
    python export_viewer.py
    python -m http.server -d docs 8000     # then open http://localhost:8000

Writes docs/data/subjects.json (the list the page's subject menu shows) and, for every
subject, a folder docs/data/sub-XXXX/ with (uint8, 0-255, so they load fast):
    t1_3t.nii.gz            3T T1 on its 1 mm grid, cropped to the head: the sharp reference
    t1_3t_edges.nii.gz      outlines of the 3T T1 (3D Canny edges), same grid, 0 or 255
    lowfield_before.nii.gz  64 mT scan, image centers lined up (no rotation)
    lowfield_after.nii.gz   64 mT scan, moved by the registration result
    results.json            that subject's row of results/all_subjects.json

The two 64 mT files contain the *same voxels* as the scanner recorded them. Only the header
(where the volume sits and which way it points) differs, so nothing is interpolated or
smoothed on our side and the real 1.6 x 1.6 x 5 mm voxels stay visible.
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from register import OUT, SUBJECT, initial_transform, load_3t_t1, load_lowfield, normalize

DOCS_DATA = Path("docs") / "data"
CROP_MARGIN_MM = 6
# Canny settings for the 3T outlines: smoothing variance (mm^2) and low/high gradient
# thresholds on the 0-1 normalized T1. Picked by eye to keep the head and brain surface and
# the ventricles without outlining every fold.
EDGE_VARIANCE_MM2 = 2.0
EDGE_THRESHOLDS = (0.08, 0.18)


def to_uint8(img):
    """normalize() gives 0-1 floats; store as 0-255 bytes (4x smaller, plenty for display)."""
    return sitk.Cast(sitk.Round(normalize(img) * 255.0), sitk.sitkUInt8)


def as_euler(tf):
    """The registration returns a one-element CompositeTransform; unwrap it."""
    if tf.GetName() == "CompositeTransform":
        tf = sitk.CompositeTransform(tf).GetNthTransform(0)
    return sitk.Euler3DTransform(tf)


def relocate(img, tf):
    """Move an image rigidly by rewriting its header instead of resampling it.

    tf maps a point of this image to where it belongs in the other scan:
    p -> R p + offset. A voxel at index i sits at origin + D S i, so after the move it sits
    at tf(origin) + (R D) S i: new origin = tf(origin), new direction = R D. Same voxels.
    """
    out = sitk.Image(img)
    R = np.array(tf.GetMatrix()).reshape(3, 3)
    D = np.array(img.GetDirection()).reshape(3, 3)
    out.SetDirection(tuple((R @ D).ravel()))
    out.SetOrigin(tf.TransformPoint(img.GetOrigin()))
    return out


def outlines(img):
    """Edges of the 3T scan as a 0/255 mask. Edges work across contrasts: a T1 outline should
    trace the same boundary in the T2 scan when the two are aligned, even though the tissue
    brightness is completely different."""
    edges = sitk.CannyEdgeDetection(normalize(img), *EDGE_THRESHOLDS, [EDGE_VARIANCE_MM2] * 3)
    return sitk.Cast(edges > 0, sitk.sitkUInt8) * 255


def crop_to_head(img, margin_mm):
    """Crop to the bounding box of the nonzero (masked) head, plus a margin."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(img > 0, sitk.sitkUInt8))
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    m = [int(np.ceil(margin_mm / s)) for s in img.GetSpacing()]
    lo = [max(0, v - mi) for v, mi in zip((x, y, z), m)]
    hi = [min(n, v + s + mi) for v, s, mi, n in zip((x, y, z), (sx, sy, sz), m, img.GetSize())]
    return sitk.RegionOfInterest(img, [h - l for h, l in zip(hi, lo)], lo)


def export_subject(subject, results):
    """Write one subject's viewer files to docs/data/<subject>/."""
    tfm = OUT / "transforms" / f"{subject}.tfm"
    out_dir = DOCS_DATA / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    low = load_lowfield(subject)
    t1 = load_3t_t1(subject)
    after = as_euler(sitk.ReadTransform(str(tfm)))
    # "Before" = the same starting guess the registration uses (image centers lined up). It
    # must be computed on the uncropped T1, because cropping moves the image center.
    before = as_euler(initial_transform(normalize(low), normalize(t1), "GEOMETRY"))

    t1_crop = crop_to_head(t1, CROP_MARGIN_MM)
    outputs = {
        "t1_3t": t1_crop,
        "t1_3t_edges": outlines(t1_crop),
        "lowfield_before": relocate(low, before),
        "lowfield_after": relocate(low, after),
    }
    total = 0
    for name, img in outputs.items():
        out = out_dir / f"{name}.nii.gz"
        data = img if name == "t1_3t_edges" else to_uint8(img)
        sitk.WriteImage(data, str(out), useCompression=True)
        total += out.stat().st_size
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir}/  ({total / 1e6:.1f} MB)")


def main():
    table = OUT / "all_subjects.json"
    if not table.exists():
        raise SystemExit(f"Missing {table}. Run `python all_subjects.py` first.")
    rows = json.loads(table.read_text())
    missing = [r["subject"] for r in rows if not (OUT / "transforms" / f"{r['subject']}.tfm").exists()]
    if missing:
        raise SystemExit(f"No saved transform for {missing}. Rerun `python all_subjects.py`.")

    for r in rows:
        export_subject(r["subject"], r)

    summary = [{
        "subject": r["subject"],
        "mi_before": r["mutual_information"]["before"],
        "mi_after": r["mutual_information"]["after"],
        "self_consistency_mean_mm": r["self_consistency"]["mean_error_mm"]["after"],
        "self_consistency_max_mm": r["self_consistency"]["max_error_mm"]["after"],
    } for r in rows]
    (DOCS_DATA / "subjects.json").write_text(
        json.dumps({"default": SUBJECT, "subjects": summary}, indent=2))
    print(f"wrote {DOCS_DATA / 'subjects.json'}")

    # Single-subject files from the earlier version of the viewer
    for old in ("t1_3t.nii.gz", "t1_3t_edges.nii.gz", "lowfield_before.nii.gz",
                "lowfield_after.nii.gz", "results.json",
                "lowfield.nii.gz", "t1_before.nii.gz", "t1_after.nii.gz"):
        (DOCS_DATA / old).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
