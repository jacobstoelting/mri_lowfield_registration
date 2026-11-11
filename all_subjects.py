"""Run the registration for every volunteer scanned at both 64 mT and 3T.

Run from the repo root after `python download_data.py`:
    python all_subjects.py

Writes to results/:
    all_subjects.json         full results per subject
    all_subjects.csv          one row per subject (the summary table)
    all_subjects_overlay.png  contact sheet: 64 mT scan with the registered 3T outlines on top
    transforms/sub-XXXX.tfm   each subject's registration result (used by export_viewer.py)

Each subject uses session 1 and the 64 mT T2 (same settings as register.py). Accuracy is
checked three ways, none of which needs a known right answer:
    self-consistency  move the 3T by a known amount, re-register, compare (register.py)
    starts agree      the two starting guesses (image centers / centers of mass) end in
                      the same place, or if not, one is clearly better (higher MI)
    contact sheet     look: the yellow 3T outlines should trace the 64 mT anatomy
"""
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from figures import _display_grid, _slices
from register import OUT, paired_subjects, run_subject

FLAG_MM = 2.0      # flag if self-consistency is worse than this...
CLOSE_MI = 0.01    # ...or if the two starts disagree by more than FLAG_MM while scoring
                   # within this much mutual information (then the choice is a coin flip)


def outline_panels(fixed, moving, tf):
    """Axial and sagittal middle slices of the 64 mT scan, with 3T edges after registration."""
    grid = _display_grid(fixed)
    low = sitk.Resample(fixed, grid, sitk.Transform(), sitk.sitkLinear, 0.0)
    t1 = sitk.Resample(moving, grid, tf, sitk.sitkLinear, 0.0)
    edges = sitk.CannyEdgeDetection(t1, 0.08, 0.18, [2.0] * 3)
    lows, edge_slices = _slices(low), _slices(edges)
    return [(lows[i], edge_slices[i]) for i in (0, 2)]  # axial, sagittal


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "transforms").mkdir(exist_ok=True)
    subjects = paired_subjects()
    rows, panels = [], []
    for subject in subjects:
        t0 = time.time()
        results, r = run_subject(subject)
        results["seconds"] = round(time.time() - t0, 1)
        sc = results["self_consistency"]
        mi_gap = abs(results["start_mi"]["GEOMETRY"] - results["start_mi"]["MOMENTS"])
        results["flagged"] = bool(sc["mean_error_mm"]["after"] > FLAG_MM
                                  or (results["starts_differ_mm"] > FLAG_MM and mi_gap < CLOSE_MI))
        rows.append(results)
        panels.append(outline_panels(r["fixed"], r["moving"], r["est_tf"]))
        sitk.WriteTransform(r["est_tf"], str(OUT / "transforms" / f"{subject}.tfm"))
        print(f"== {subject}: MI {results['mutual_information']['before']} -> "
              f"{results['mutual_information']['after']}, self-consistency "
              f"{sc['mean_error_mm']['after']} mm, starts differ {results['starts_differ_mm']} mm "
              f"({results['seconds']} s)\n")

    (OUT / "all_subjects.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "all_subjects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "mi_before", "mi_after", "self_consistency_mean_mm",
                    "self_consistency_max_mm", "starts_differ_mm", "mi_geometry_start",
                    "mi_moments_start", "start_used", "flagged"])
        for r in rows:
            sc = r["self_consistency"]
            w.writerow([r["subject"], r["mutual_information"]["before"],
                        r["mutual_information"]["after"], sc["mean_error_mm"]["after"],
                        sc["max_error_mm"]["after"], r["starts_differ_mm"],
                        r["start_mi"]["GEOMETRY"], r["start_mi"]["MOMENTS"], r["start_used"],
                        r["flagged"]])

    # Contact sheet: 2 panels (axial, sagittal) per subject, 4 subjects per row.
    per_row = 4
    nrows = int(np.ceil(len(subjects) / per_row))
    fig, ax = plt.subplots(nrows, per_row * 2, figsize=(per_row * 4.2, nrows * 2.6))
    for a in ax.ravel():
        a.axis("off")
    for k, (subject, pair) in enumerate(zip(subjects, panels)):
        row, col = divmod(k, per_row)
        for j, (img, edge) in enumerate(pair):
            a = ax[row, col * 2 + j]
            a.imshow(img, cmap="gray", vmin=0, vmax=1)
            a.imshow(np.ma.masked_where(edge == 0, edge), cmap="autumn_r", interpolation="nearest")
        r = rows[k]
        ax[row, col * 2].set_title(
            f"{subject}: {r['self_consistency']['mean_error_mm']['after']} mm"
            + ("  FLAGGED" if r["flagged"] else ""), fontsize=9, loc="left")
    fig.suptitle("64 mT T2 (gray) with the registered 3T T1 outlines (yellow), axial and sagittal. "
                 "Title: self-consistency error.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "all_subjects_overlay.png", dpi=90)
    plt.close(fig)

    sc = np.array([r["self_consistency"]["mean_error_mm"]["after"] for r in rows])
    print(f"Self-consistency over {len(rows)} subjects: median {np.median(sc):.2f} mm, "
          f"max {sc.max():.2f} mm; flagged: {[r['subject'] for r in rows if r['flagged']] or 'none'}")


if __name__ == "__main__":
    main()
