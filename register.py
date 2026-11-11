"""Align a 3T brain MRI to a real 64 mT low-field scan of the same person with SimpleITK.

Data: "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research"
(van den Broek, Lena & Webb, LUMC; Zenodo 10.5281/zenodo.20281403, MIT license).
Get it with `python download_data.py`.

Pipeline
  1. Load the 64 mT scan (Hyperfine Swoop, 1.6 x 1.6 x 5 mm voxels) as the *fixed* image and
     the 3T T1 (Philips, measured at 1 mm isotropic) as the *moving* image.
  2. Mask the background of each scan and normalize intensities.
  3. Start by lining up the image centers (or centers of mass), then rigidly register
     3T -> 64 mT by maximizing mutual information (works across T1 vs T2 contrast).
  4. Real data has no known answer, so measure accuracy with a self-consistency test:
     move the 3T scan by a known rotation + shift, register again from scratch, and check
     that the second answer differs from the first by exactly that known movement.
"""
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA = Path("data")
OUT = Path("results")
LEIDEN = DATA / "leiden" / "v5 Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research" / "Data"

# Which volunteer and which scans. The 64 mT and 3T folders share subject IDs (same person).
# This subject drives the figures and the web viewer; all_subjects.py runs every pair.
SUBJECT = "sub-0035"
SESSION = "ses-01"
LOWFIELD_CONTRAST = "T2w"  # 64 mT scan used as the fixed image: "T2w", "FLAIR" or "T1w"


def lowfield_path(subject=SUBJECT, session=SESSION, contrast=LOWFIELD_CONTRAST):
    return (LEIDEN / "64mT data" / subject / session / "anat"
            / f"{subject}_{session}_run-1_{contrast}.nii.gz")


def t1_3t_path(subject=SUBJECT):
    return LEIDEN / "3T data" / subject / "anat" / f"{subject}_acq-highres_T1w.nii.gz"


def paired_subjects():
    """The volunteers scanned on both machines (the 3T folder only has those)."""
    return sorted(p.name for p in (LEIDEN / "3T data").glob("sub-*") if p.is_dir())


T1_SPACING_MM = 1.0  # the 3T T1 was measured at 1 mm; the file is interpolated to 0.49 mm in-plane

# Known movement for the self-consistency test (and for simulate_low_field)
TRUE_ROTATION_DEG = (6.0, -4.0, 9.0)  # around x, y, z (degrees)
TRUE_SHIFT_MM = (7.0, -12.0, 5.0)     # along x, y, z (mm)

# Only used by simulate_low_field (kept for the robustness sweep)
LOWFIELD_SPACING = (1.6, 1.6, 5.0)  # mm
NOISE_FRACTION = 0.06               # noise std as a fraction of the 99th-percentile intensity
SEED = 0


def euler(rot_deg, shift_mm, center):
    t = sitk.Euler3DTransform()
    t.SetCenter(center)
    t.SetRotation(*[math.radians(a) for a in rot_deg])
    t.SetTranslation(shift_mm)
    return t


def image_center(img):
    """Physical (mm) coordinates of the middle of the image."""
    mid = [(n - 1) / 2.0 for n in img.GetSize()]
    return img.TransformContinuousIndexToPhysicalPoint(mid)


def mask_background(img, mask_source=None):
    """Zero everything outside the head.

    The head mask comes from `mask_source` (e.g. a separate image with cleaner background)
    or, by default, from a smoothed copy of the image itself: Otsu threshold, then close
    small gaps and fill holes so dark tissue inside the head (CSF) is kept.
    """
    if mask_source is None:
        mask_source = sitk.SmoothingRecursiveGaussian(sitk.Cast(img, sitk.sitkFloat32), 2.0)
    head = sitk.OtsuThreshold(sitk.Cast(mask_source, sitk.sitkFloat32), 0, 1)
    head = sitk.BinaryMorphologicalClosing(head, [3, 3, 3])
    head = sitk.BinaryFillhole(head)
    return sitk.Cast(img, sitk.sitkFloat32) * sitk.Cast(head, sitk.sitkFloat32)


def resample_isotropic(img, spacing):
    """Resample onto a grid with the same orientation and extent but cubic voxels."""
    size = [int(round(n * s / spacing)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing((spacing,) * 3)
    grid.SetDirection(img.GetDirection())
    # keep the physical corner of the volume where it was (voxel centers shift by half a voxel)
    corner = np.array(img.TransformContinuousIndexToPhysicalPoint([-0.5] * 3))
    d = np.array(img.GetDirection()).reshape(3, 3)
    grid.SetOrigin(tuple(corner + d @ np.full(3, spacing / 2.0)))
    return sitk.Resample(img, grid, sitk.Transform(), sitk.sitkLinear, 0.0)


def load_lowfield(subject=SUBJECT, session=SESSION, contrast=LOWFIELD_CONTRAST):
    """The real 64 mT scan with background masked (float32, original 1.6 x 1.6 x 5 mm grid)."""
    return mask_background(sitk.ReadImage(str(lowfield_path(subject, session, contrast)),
                                          sitk.sitkFloat32))


def load_3t_t1(subject=SUBJECT):
    """The 3T T1 on a 1 mm isotropic grid (its true acquired resolution), background masked."""
    t1 = sitk.ReadImage(str(t1_3t_path(subject)), sitk.sitkFloat32)
    return mask_background(resample_isotropic(t1, T1_SPACING_MM))


def simulate_low_field(flair, true_tf, rng):
    """Resample a scan onto a coarse grid while applying a known head movement, then add noise.

    Not used in the main pipeline any more (we have real 64 mT data); kept for the
    robustness sweep, where a known answer is needed.
    """
    size = [int(round(n * s / ls)) for n, s, ls in
            zip(flair.GetSize(), flair.GetSpacing(), LOWFIELD_SPACING)]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(LOWFIELD_SPACING)
    grid.SetOrigin(flair.GetOrigin())
    grid.SetDirection(flair.GetDirection())
    # Smooth first so the coarse voxels average their area instead of aliasing (in-plane only).
    variance = [(ls / 2.0) ** 2 for ls in LOWFIELD_SPACING[:2]] + [0.0]
    smooth = sitk.DiscreteGaussian(sitk.Cast(flair, sitk.sitkFloat32), variance)
    moved = sitk.Resample(smooth, grid, true_tf, sitk.sitkLinear, 0.0)

    # Rician noise: MRI magnitude images are |signal + complex Gaussian noise|.
    a = sitk.GetArrayFromImage(moved)
    sd = NOISE_FRACTION * np.percentile(a, 99)
    noisy = np.sqrt((a + rng.normal(0, sd, a.shape)) ** 2 + rng.normal(0, sd, a.shape) ** 2)
    out = sitk.GetImageFromArray(noisy.astype(np.float32))
    out.CopyInformation(moved)
    return out


def normalize(img):
    """Clip to the 1st-99th percentile of the head and rescale to 0-1 so both scans share a range."""
    a = sitk.GetArrayFromImage(img)
    lo, hi = np.percentile(a[a > 0], [1, 99])
    out = sitk.GetImageFromArray(np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32))
    out.CopyInformation(img)
    return out


MAX_ITERATIONS = 2000  # per resolution level; 400 cut some subjects off before convergence


def register(fixed, moving, init, shrink=(4, 2, 1), smooth_mm=(2, 1, 0), verbose=True,
             iterations=None):
    """Rigid (rotation + translation) registration that maximizes mutual information."""
    reg = sitk.ImageRegistrationMethod()
    # Mutual information works across contrasts (T1 vs T2): it rewards a consistent
    # mapping between the two intensity patterns, not equal intensities.
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.NONE)  # use every voxel; random subsets made it jitter
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iterations or MAX_ITERATIONS,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
    # Make 1 radian of rotation and 1 mm of shift comparable for the optimizer.
    reg.SetOptimizerScalesFromPhysicalShift()
    # Coarse-to-fine: solve on blurry, shrunken images first, then refine.
    reg.SetShrinkFactorsPerLevel(list(shrink))
    reg.SetSmoothingSigmasPerLevel(list(smooth_mm))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)

    history = []
    reg.AddCommand(sitk.sitkIterationEvent, lambda: history.append(reg.GetMetricValue()))
    tf = reg.Execute(fixed, moving)
    if verbose:
        print(f"  optimizer stopped: {reg.GetOptimizerStopConditionDescription()}")
    return tf, history


def mutual_information(fixed, moving, tf):
    """Mattes MI between the two scans for a given transform (SimpleITK reports it negated)."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(tf)
    return -reg.MetricEvaluate(fixed, moving)


def registration_error_mm(fixed, est_tf, true_tf):
    """Mean and max distance (mm) between where the estimate and the truth send each head voxel."""
    a = sitk.GetArrayViewFromImage(fixed)
    idx = np.argwhere(a > 0.2)[::50]  # head voxels (z, y, x), subsampled
    d = []
    for z, y, x in idx:
        p = fixed.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        d.append(np.linalg.norm(np.subtract(est_tf.TransformPoint(p), true_tf.TransformPoint(p))))
    return float(np.mean(d)), float(np.max(d))


def initial_transform(fixed, moving, how):
    """Rotation-free starting guess that lines up the image centers ("GEOMETRY") or the
    intensity centers of mass ("MOMENTS"). Needed because the two scanners' coordinate
    systems are unrelated (here they are over 20 cm apart)."""
    mode = getattr(sitk.CenteredTransformInitializerFilter, how)
    return sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), mode)


def register_best(fixed, moving):
    """Try both starting guesses and keep the result with the higher mutual information.

    Also returns how far apart (mean mm over the head) the two answers are. If two different
    starting points end at the same place, the optimizer is less likely to be stuck in a
    wrong local optimum."""
    runs = []
    for how in ("GEOMETRY", "MOMENTS"):
        print(f" start: {how}")
        tf, history = register(fixed, moving, initial_transform(fixed, moving, how))
        mi = mutual_information(fixed, moving, tf)
        print(f"  final MI {mi:.4f}")
        runs.append((mi, how, tf, history))
    starts_differ_mm = registration_error_mm(fixed, runs[0][2], runs[1][2])[0]
    start_mi = {how: round(mi, 4) for mi, how, _, _ in runs}
    mi, how, tf, history = max(runs, key=lambda r: r[0])
    return mi, how, tf, history, starts_differ_mm, start_mi


def run_subject(subject=SUBJECT, session=SESSION, contrast=LOWFIELD_CONTRAST):
    """Register one volunteer's 3T T1 to their 64 mT scan and measure the result.

    Returns (results dict, images and transforms needed for figures/outputs)."""
    low = load_lowfield(subject, session, contrast)
    t1 = load_3t_t1(subject)
    fixed, moving = normalize(low), normalize(t1)
    print(f"{subject}: 64 mT {contrast} {low.GetSize()} @ "
          f"{tuple(round(s, 2) for s in low.GetSpacing())} mm (fixed) <- 3T T1 "
          f"{t1.GetSize()} @ {t1.GetSpacing()} mm (moving)")

    identity = sitk.Euler3DTransform()
    start = initial_transform(fixed, moving, "GEOMETRY")  # "before": centers lined up

    print("Register 3T T1 -> 64 mT")
    after_mi, init_used, est_tf, history, starts_differ, start_mi = register_best(fixed, moving)

    # Self-consistency: move the 3T scan by a known amount P and register again from scratch.
    # moved(x) = t1(P(x)), so if est maps low-field points to T1 points, the new answer
    # should be est2 = P^-1 o est, i.e. P o est2 should equal est.
    print("Self-consistency test: known extra movement of the 3T scan")
    P = euler(TRUE_ROTATION_DEG, TRUE_SHIFT_MM, image_center(t1))
    moved = normalize(sitk.Resample(t1, t1, P, sitk.sitkLinear, 0.0))
    est2 = register_best(fixed, moved)[2]
    sc_before = registration_error_mm(fixed, sitk.CompositeTransform([P, est_tf]), est_tf)
    sc_after = registration_error_mm(fixed, sitk.CompositeTransform([P, est2]), est_tf)

    euler_est = sitk.Euler3DTransform(sitk.CompositeTransform(est_tf).GetNthTransform(0))
    results = {
        "subject": subject,
        "session": session,
        "fixed": f"64 mT {contrast} (Hyperfine Swoop)",
        "moving": "3T T1 (Philips Achieva), 1 mm",
        "scanner_frame_offset_mm": round(float(np.linalg.norm(
            np.subtract(image_center(low), image_center(t1)))), 1),
        "mutual_information": {
            "raw_scanner_coordinates": round(mutual_information(fixed, moving, identity), 4),
            "before": round(mutual_information(fixed, moving, start), 4),
            "after": round(after_mi, 4),
        },
        "start_used": init_used,
        "start_mi": start_mi,
        "starts_differ_mm": round(starts_differ, 2),
        "estimated_rotation_deg": [round(math.degrees(a), 2) for a in
                                   (euler_est.GetAngleX(), euler_est.GetAngleY(), euler_est.GetAngleZ())],
        "self_consistency": {
            "known_rotation_deg": TRUE_ROTATION_DEG,
            "known_shift_mm": TRUE_SHIFT_MM,
            "mean_error_mm": {"before": round(sc_before[0], 2), "after": round(sc_after[0], 2)},
            "max_error_mm": {"before": round(sc_before[1], 2), "after": round(sc_after[1], 2)},
        },
        "iterations": len(history),
    }
    return results, dict(low=low, t1=t1, fixed=fixed, moving=moving, start=start,
                         est_tf=est_tf, history=history)


def main():
    OUT.mkdir(exist_ok=True)
    results, r = run_subject()
    low, t1, est_tf, history = r["low"], r["t1"], r["est_tf"], r["history"]

    sitk.WriteImage(low, str(OUT / "lowfield.nii.gz"))
    aligned = sitk.Resample(t1, low, est_tf, sitk.sitkLinear, 0.0)
    sitk.WriteImage(aligned, str(OUT / "t1_aligned_to_lowfield.nii.gz"))
    sitk.WriteTransform(est_tf, str(OUT / "t1_to_lowfield.tfm"))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    np.savetxt(OUT / "metric_history.txt", history)
    print(json.dumps(results, indent=2))

    from figures import save_figures
    save_figures(r["fixed"], r["moving"], r["start"], est_tf, history, results, OUT)


if __name__ == "__main__":
    main()
