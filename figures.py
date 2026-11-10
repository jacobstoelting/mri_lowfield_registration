"""Before/after overlays and the optimizer curve."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def _display_grid(fixed, spacing=1.5, extent_mm=(200, 230, 160)):
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


def _checker(f, m, n=8):
    yy, xx = np.indices(f.shape)
    tiles = ((yy * n // f.shape[0]) + (xx * n // f.shape[1])) % 2 == 0
    return np.where(tiles, f, m)


def save_figures(fixed, moving, before_tf, after_tf, history, results, out):
    # View everything on a straight, isotropic grid so the 5 mm slices don't look squashed.
    iso = sitk.Resample(fixed, _display_grid(fixed), sitk.Transform(), sitk.sitkLinear, 0.0)
    rows = [("Before", before_tf), ("After", after_tf)]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8.5))
    fs = _slices(iso)
    for r, (name, tf) in enumerate(rows):
        ms = _slices(sitk.Resample(moving, iso, tf, sitk.sitkLinear, 0.0))
        for c, (f, m, view) in enumerate(zip(fs, ms, ["axial", "coronal", "sagittal"])):
            rgb = np.dstack([f, m, np.zeros_like(f)])  # red = low-field, green = 3T T1
            ax[r, c].imshow(np.clip(rgb, 0, 1))
            ax[r, c].set_title(f"{name}: {view}")
            ax[r, c].axis("off")
    mi, err = results["mutual_information"], results["mean_error_mm"]
    fig.suptitle(f"Red = simulated low-field FLAIR, green = 3T T1 (yellow where both are bright)\n"
                 f"Mutual information {mi['before']} -> {mi['after']}   |   "
                 f"mean error {err['before']} mm -> {err['after']} mm")
    fig.tight_layout()
    fig.savefig(out / "overlay_before_after.png", dpi=110)
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
