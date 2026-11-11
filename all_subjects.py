"""Run the registration for every volunteer scanned at both 64 mT and 3T.

Run from the repo root after `python download_data.py`:
    python all_subjects.py           # only registrations not already in all_subjects.json
    python all_subjects.py --all     # redo everything (after changing the pipeline)

Writes to results/:
    all_subjects.json         full results per subject
    all_subjects.csv          one row per subject (the summary table)
    all_subjects_overlay.png  contact sheet: 64 mT scan with the registered 3T outlines on top
    transforms/sub-XXXX_<contrast>.tfm   each registration result (used by export_viewer.py)

Each subject uses session 1. Every 64 mT contrast (T2w, FLAIR, T1w) is registered to the same
3T T1, so differences between contrasts come from the 64 mT scan alone. The contact sheet and
the README table use the T2 (the main result). Accuracy is
checked three ways, none of which needs a known right answer:
    self-consistency  move the 3T by a known amount, re-register, compare (register.py)
    starts agree      the two starting guesses (image centers / centers of mass) end in
                      the same place, or if not, one is clearly better (higher MI)
    contact sheet     look: the yellow 3T outlines should trace the 64 mT anatomy
"""
import csv
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from figures import _display_grid, _slices
from register import OUT, paired_subjects, run_subject

CONTRASTS = ("T2w", "FLAIR", "T1w")  # 64 mT scans used as the fixed image; T2w is the main one

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
    # Keep finished registrations (each takes up to ~10 minutes) unless asked to redo them.
    table = OUT / "all_subjects.json"
    done = {} if "--all" in sys.argv or not table.exists() else {
        (r["subject"], r.get("contrast", "T2w")): r for r in json.loads(table.read_text())}
    rows, panels = [], []
    for subject, contrast in [(s, c) for s in subjects for c in CONTRASTS]:
        if (subject, contrast) in done and (OUT / "transforms" / f"{subject}_{contrast}.tfm").exists():
            rows.append({**done[(subject, contrast)], "contrast": contrast})
            continue
        t0 = time.time()
        results, r = run_subject(subject, contrast=contrast)
        results["contrast"] = contrast
        results["seconds"] = round(time.time() - t0, 1)
        sc = results["self_consistency"]
        mi_gap = abs(results["start_mi"]["GEOMETRY"] - results["start_mi"]["MOMENTS"])
        results["flagged"] = bool(not results["converged"]
                                  or sc["mean_error_mm"]["after"] > FLAG_MM
                                  or (results["starts_differ_mm"] > FLAG_MM and mi_gap < CLOSE_MI))
        rows.append(results)
        if contrast == "T2w":
            panels.append(outline_panels(r["fixed"], r["moving"], r["est_tf"]))
        sitk.WriteTransform(r["est_tf"], str(OUT / "transforms" / f"{subject}_{contrast}.tfm"))
        print(f"== {subject} {contrast}: MI {results['mutual_information']['before']} -> "
              f"{results['mutual_information']['after']}, self-consistency "
              f"{sc['mean_error_mm']['after']} mm, starts differ {results['starts_differ_mm']} mm "
              f"({results['seconds']} s)\n")

    (OUT / "all_subjects.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "all_subjects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "contrast", "mi_before", "mi_after", "self_consistency_mean_mm",
                    "self_consistency_max_mm", "starts_differ_mm", "mi_geometry_start",
                    "mi_moments_start", "start_used", "flagged"])
        for r in rows:
            sc = r["self_consistency"]
            w.writerow([r["subject"], r["contrast"], r["mutual_information"]["before"],
                        r["mutual_information"]["after"], sc["mean_error_mm"]["after"],
                        sc["max_error_mm"]["after"], r["starts_differ_mm"],
                        r["start_mi"]["GEOMETRY"], r["start_mi"]["MOMENTS"], r["start_used"],
                        r["flagged"]])

    for contrast in CONTRASTS:
        sel = [r for r in rows if r["contrast"] == contrast]
        sc = np.array([r["self_consistency"]["mean_error_mm"]["after"] for r in sel])
        print(f"{contrast}: self-consistency over {len(sel)} subjects: median {np.median(sc):.2f} mm, "
              f"max {sc.max():.2f} mm; flagged: {[r['subject'] for r in sel if r['flagged']] or 'none'}")

    if len(panels) < len(subjects):
        print("T2 registrations were reused, so the contact sheet was left as it is.")
        return
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
        r = [x for x in rows if x["contrast"] == "T2w"][k]
        ax[row, col * 2].set_title(
            f"{subject}: {r['self_consistency']['mean_error_mm']['after']} mm"
            + ("  FLAGGED" if r["flagged"] else ""), fontsize=9, loc="left")
    fig.suptitle("64 mT T2 (gray) with the registered 3T T1 outlines (yellow), axial and sagittal. "
                 "Title: self-consistency error.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "all_subjects_overlay.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
