"""Before/after outline figures, the optimizer curve, and the all-subjects accuracy chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def _display_grid(fixed, spacing=1.5, extent_mm=(200, 230, 210)):
    """An axis-aligned grid (rows/columns follow the LPS axes) centered on the fixed scan.

    The low-field scan's slices are tilted ~23 degrees, so viewing it on its own grid looks
    crooked. SimpleITK works in LPS physical coordinates (x -> patient Left, y -> Posterior,
    z -> Superior), so an identity direction matrix gives standard views.
    """
    mid = [(n - 1) / 2.0 for n in fixed.GetSize()]
    center = np.array(fixed.TransformContinuousIndexToPhysicalPoint(mid))
    size = [int(e / spacing) for e in extent_mm]
    origin = center - (np.array(size) - 1) * spacing / 2.0
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing((spacing,) * 3)
    grid.SetOrigin(tuple(origin))
    return grid


def _slices(img):
    """Middle axial, coronal and sagittal slices, flipped so the head is upright.

    Array order is [z, y, x]. Axial: anterior at top, patient's right on screen left
    (the radiology convention). Coronal and sagittal: superior at top.
    """
    a = sitk.GetArrayFromImage(img)
    z, y, x = (n // 2 for n in a.shape)
    return [a[z], a[::-1, y, :], a[::-1, :, x]]


_EDGE_CMAP = matplotlib.colors.ListedColormap(["#ffd400"])  # the viewer's outline yellow


def _checker(f, m, n=8):
    yy, xx = np.indices(f.shape)
    tiles = ((yy * n // f.shape[0]) + (xx * n // f.shape[1])) % 2 == 0
    return np.where(tiles, f, m)


def save_figures(fixed, moving, before_tf, after_tf, history, results, out):
    # View everything on a straight, isotropic grid so the 5 mm slices don't look squashed.
    iso = sitk.Resample(fixed, _display_grid(fixed, spacing=1.0), sitk.Transform(), sitk.sitkLinear, 0.0)
    rows = [("Before", before_tf), ("After", after_tf)]
    fs = _slices(iso)

    # Same look as the viewer: the 64 mT scan in grayscale with the 3T outlines (3D Canny
    # edges, the viewer's settings) in yellow. Edges work across contrasts, so they show the
    # alignment without asking the reader to compare T1 and T2 brightness.
    from export_viewer import outlines
    edges = sitk.Cast(outlines(moving), sitk.sitkFloat32) / 255.0
    fig, ax = plt.subplots(2, 3, figsize=(12, 8.2), facecolor="black")
    for r, (name, tf) in enumerate(rows):
        # Linear resampling then a low threshold keeps the 1-voxel edges unbroken on the
        # 1 mm display grid.
        es = _slices(sitk.Resample(edges, iso, tf, sitk.sitkLinear, 0.0))
        for c, (f, e, view) in enumerate(zip(fs, es, ["axial", "coronal", "sagittal"])):
            ax[r, c].imshow(f, cmap="gray", vmin=0, vmax=1)
            ax[r, c].imshow(np.ma.masked_less(e, 0.2), cmap=_EDGE_CMAP, vmin=0, vmax=1,
                            interpolation="nearest")
            ax[r, c].set_title(f"{name}: {view}", color="white")
            ax[r, c].axis("off")
    # Accuracy is reported for all subjects (accuracy_summary.png and the README table), not
    # here: one subject's number next to the headline figure would read as the headline.
    mi = results["mutual_information"]
    fig.suptitle(f"{results['subject']}: {results['fixed']} in gray, outlines of the "
                 f"{results['moving']} in yellow\n"
                 f"Before = image centers lined up.  Mutual information {mi['before']} -> {mi['after']}",
                 color="white")
    fig.tight_layout()
    fig.savefig(out / "overlay_before_after.png", dpi=110, facecolor="black")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for c, (name, tf) in enumerate(rows):
        f = fs[0]
        m = _slices(sitk.Resample(moving, iso, tf, sitk.sitkLinear, 0.0))[0]
        ax[c].imshow(_checker(f, m), cmap="gray")
        ax[c].set_title(f"{name}: checkerboard (axial)")
        ax[c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "checkerboard.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(-np.array(history))
    ax.set_xlabel("Optimizer iteration (all 3 resolution levels)")
    ax.set_ylabel("Mutual information")
    ax.set_title("Registration progress")
    fig.tight_layout()
    fig.savefig(out / "metric_history.png", dpi=110)
    plt.close(fig)


# Dark, to match the before/after figure. Colorblind-safe pair validated on black
# (CVD Delta E 26.8, both >= 3:1 contrast), plus ink colors for text and the grid.
SELF_COLOR, LOOP_COLOR = "#3987e5", "#d95926"
INK, INK_MUTED, GRID, SURFACE = "#f0efec", "#a8a79f", "#2e2e2c", "#000000"
VOXEL_IN_PLANE_MM = 1.6


def save_accuracy_summary(out):
    """Both accuracy checks for every subject (64 mT T2 -> 3T T1), against the 64 mT voxel.

    Reads results/all_subjects.csv (all_subjects.py) and results/consistency_loop.json
    (consistency_loop.py), and writes results/accuracy_summary.png.
    """
    import csv
    import json
    self_c = {r["subject"]: float(r["self_consistency_mean_mm"])
              for r in csv.DictReader(open(out / "all_subjects.csv")) if r["contrast"] == "T2w"}
    loop = {r["subject"]: r["loop_error_mm"]["mean"]
            for r in json.loads((out / "consistency_loop.json").read_text())}
    subjects = sorted(loop, key=loop.get)  # best at the bottom, worst at the top
    y = np.arange(len(subjects))
    s = np.array([self_c[k] for k in subjects])
    lp = np.array([loop[k] for k in subjects])

    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hlines(y, np.minimum(s, lp), np.maximum(s, lp), color=GRID, lw=2, zorder=1)
    ax.axvline(VOXEL_IN_PLANE_MM, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
    # Label in the empty lower-right area: only the worst subjects reach past the line.
    ax.text(VOXEL_IN_PLANE_MM + 0.07, 4, "one 64 mT voxel, in-plane\n(1.6 mm; slices are 5 mm)",
            color=INK_MUTED, fontsize=9, va="center")
    # Shape as well as color, so the two checks don't rely on color alone.
    ax.scatter(s, y, s=55, color=SELF_COLOR, marker="o", edgecolor=SURFACE, lw=1.5, zorder=3,
               label=f"Self-consistency (optimistic): median {np.median(s):.2f} mm")
    ax.scatter(lp, y, s=55, color=LOOP_COLOR, marker="s", edgecolor=SURFACE, lw=1.5, zorder=3,
               label=f"Consistency loop (stricter): median {np.median(lp):.2f} mm")
    for i in np.flatnonzero(lp > VOXEL_IN_PLANE_MM):  # label the ones past the voxel size
        ax.text(lp[i] + 0.1, i, f"{lp[i]:.2f} mm", color=INK, fontsize=9, va="center")

    ax.set_yticks(y, subjects)
    ax.set_xlim(0, max(lp.max(), s.max()) + 0.7)
    ax.set_xlabel("Mean error over the head (mm)", color=INK)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Registration error, 64 mT T2 to 3T T1, all 11 volunteers",
                 color=INK, loc="left", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "accuracy_summary.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path
    save_accuracy_summary(Path(__file__).parent / "results")
