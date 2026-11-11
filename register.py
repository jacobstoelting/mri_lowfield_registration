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


def three_t_path(subject=SUBJECT, contrast="T1w", acq="highres"):
    """A 3T scan. acq="highres" is the clinical protocol, "lowres" the one acquired at about
    the 64 mT voxel size (1.38 x 1.38 x 5 mm)."""
    return LEIDEN / "3T data" / subject / "anat" / f"{subject}_acq-{acq}_{contrast}.nii.gz"


def t1_3t_path(subject=SUBJECT):
    return three_t_path(subject, "T1w", "highres")


def paired_subjects():
    """The volunteers scanned on both machines (the 3T folder only has those)."""
    return sorted(p.name for p in (LEIDEN / "3T data").glob("sub-*") if p.is_dir())


def lowfield_sessions(subject=SUBJECT):
    """The subject's 64 mT sessions, e.g. ["ses-01", "ses-02"]."""
    return sorted(p.name for p in (LEIDEN / "64mT data" / subject).glob("ses-*") if p.is_dir())


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


def resample_spacing(img, spacing):
    """Resample onto a grid with the same orientation and extent but the given voxel size (mm,
    one number for cubic voxels or one per axis)."""
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), img.GetSpacing(), spacing)]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(tuple(spacing))
    grid.SetDirection(img.GetDirection())
    # keep the physical corner of the volume where it was (voxel centers shift by half a voxel)
    corner = np.array(img.TransformContinuousIndexToPhysicalPoint([-0.5] * 3))
    d = np.array(img.GetDirection()).reshape(3, 3)
    grid.SetOrigin(tuple(corner + d @ (spacing / 2.0)))
    return sitk.Resample(img, grid, sitk.Transform(), sitk.sitkLinear, 0.0)


def resample_isotropic(img, spacing):
    return resample_spacing(img, spacing)


def load_lowfield(subject=SUBJECT, session=SESSION, contrast=LOWFIELD_CONTRAST):
    """The real 64 mT scan with background masked (float32, original 1.6 x 1.6 x 5 mm grid)."""
    return mask_background(sitk.ReadImage(str(lowfield_path(subject, session, contrast)),
                                          sitk.sitkFloat32))


def load_3t_t1(subject=SUBJECT):
    """The 3T T1 on a 1 mm isotropic grid (its true acquired resolution), background masked."""
    t1 = sitk.ReadImage(str(t1_3t_path(subject)), sitk.sitkFloat32)
    return mask_background(resample_isotropic(t1, T1_SPACING_MM))


# The 3T high-res T2 file has 0.22 mm in-plane voxels (interpolated) and 3 mm slices. Nothing
# finer than the 1 mm T1 it is compared with is needed, so it is resampled to 1 x 1 x 3 mm.
T2_3T_SPACING_MM = (1.0, 1.0, 3.0)


def load_3t(subject=SUBJECT, contrast="T1w", acq="highres"):
    """Any 3T scan, background masked. The high-res T1 and T2 are brought to a manageable grid;
    the other scans are used as stored."""
    if (contrast, acq) == ("T1w", "highres"):
        return load_3t_t1(subject)
    img = sitk.ReadImage(str(three_t_path(subject, contrast, acq)), sitk.sitkFloat32)
    if (contrast, acq) == ("T2w", "highres"):
        img = resample_spacing(img, T2_3T_SPACING_MM)
    return mask_background(img)


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


# Per resolution level. The full-resolution level should stop because its step shrank below
# minStep (converged), not because it ran out: 400 cut several subjects off, and 2000 still cut
# off sub-0011, which creeps slowly along a nearly flat direction. Coarse levels are cheap, so a
# generous limit costs little time; runs that converge stop early anyway.
MAX_ITERATIONS = 20000


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
    stop = reg.GetOptimizerStopConditionDescription()
    if verbose:
        print(f"  optimizer stopped: {stop}")
    # The stop condition is the full-resolution level's: converged means its step shrank to minStep.
    converged = "Step too small" in stop
    return tf, history, converged


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
        tf, history, converged = register(fixed, moving, initial_transform(fixed, moving, how))
        mi = mutual_information(fixed, moving, tf)
        print(f"  final MI {mi:.4f}")
        runs.append((mi, how, tf, history, converged))
    starts_differ_mm = registration_error_mm(fixed, runs[0][2], runs[1][2])[0]
    start_mi = {how: round(mi, 4) for mi, how, _, _, _ in runs}
    mi, how, tf, history, converged = max(runs, key=lambda r: r[0])
    return mi, how, tf, history, starts_differ_mm, start_mi, converged


def register_images(fixed_img, moving_img):
    """Normalize two masked scans and register moving -> fixed from both starting guesses.
    Returns (normalized fixed, normalized moving, transform mapping fixed points to moving
    points, final MI, whether it converged)."""
    fixed, moving = normalize(fixed_img), normalize(moving_img)
    mi, _, tf, _, _, _, converged = register_best(fixed, moving)
    if not converged:
        print("  WARNING: did not converge")
    return fixed, moving, tf, mi, converged


def run_subject(subject=SUBJECT, session=SESSION, contrast=LOWFIELD_CONTRAST):
    """Register one volunteer's 3T T1 to their 64 mT scan and measure the result.

    Returns (results dict, images and transforms needed for figures/outputs)."""
    results, r = run_pair(load_lowfield(subject, session, contrast), load_3t_t1(subject),
                          f"64 mT {contrast} (Hyperfine Swoop)")
    return {"subject": subject, "session": session, **results}, r


def run_pair(low, t1, fixed_name):
    """Register the 3T T1 (moving) to a coarse scan (fixed, e.g. the 64 mT scan) and measure
    the result: mutual information, the two starting guesses, and self-consistency."""
    fixed, moving = normalize(low), normalize(t1)
    print(f"{fixed_name} {low.GetSize()} @ "
          f"{tuple(round(s, 2) for s in low.GetSpacing())} mm (fixed) <- 3T T1 "
          f"{t1.GetSize()} @ {t1.GetSpacing()} mm (moving)")

    identity = sitk.Euler3DTransform()
    start = initial_transform(fixed, moving, "GEOMETRY")  # "before": centers lined up

    print(f"Register 3T T1 -> {fixed_name}")
    after_mi, init_used, est_tf, history, starts_differ, start_mi, converged = \
        register_best(fixed, moving)

    # Self-consistency: move the 3T scan by a known amount P and register again from scratch.
    # moved(x) = t1(P(x)), so if est maps low-field points to T1 points, the new answer
    # should be est2 = P^-1 o est, i.e. P o est2 should equal est.
    print("Self-consistency test: known extra movement of the 3T scan")
    P = euler(TRUE_ROTATION_DEG, TRUE_SHIFT_MM, image_center(t1))
    moved = normalize(sitk.Resample(t1, t1, P, sitk.sitkLinear, 0.0))
    _, _, est2, *_, converged2 = register_best(fixed, moved)
    sc_before = registration_error_mm(fixed, sitk.CompositeTransform([P, est_tf]), est_tf)
    sc_after = registration_error_mm(fixed, sitk.CompositeTransform([P, est2]), est_tf)

    euler_est = sitk.Euler3DTransform(sitk.CompositeTransform(est_tf).GetNthTransform(0))
    results = {
        "fixed": fixed_name,
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
        "converged": bool(converged and converged2),
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
