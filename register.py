"""Align a prior high-field (3T) brain MRI to a new low-field-style scan with SimpleITK.

Pipeline
  1. Load the 3T T1-weighted scan (the "prior" scan) and a FLAIR from the same session.
  2. Turn the FLAIR into a simulated low-field scan: coarser voxels, more noise,
     and a known head movement (so we know the right answer).
  3. Normalize intensities, then rigidly register T1 -> low-field using mutual information.
  4. Report how well they line up before and after (mutual information + error in mm).

How we know the right answer: the T1 and FLAIR were taken minutes apart, so the person
barely moved. We first register the T1 to the full-quality FLAIR to measure that small
"same-visit" offset. The correct answer for the low-field scan is that offset plus the
movement we added on purpose.
"""
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA = Path("data")
OUT = Path("results")
T1_PATH = DATA / "sub-010001_ses-02_acq-mp2rage_T1w.nii.gz"
INV2_PATH = DATA / "sub-010001_ses-02_inv-2_mp2rage.nii.gz"
FLAIR_PATH = DATA / "sub-010001_ses-02_acq-lowres_FLAIR.nii.gz"

# Simulated low-field settings (Hyperfine-style 64 mT scans are roughly 1.5 x 1.5 x 5 mm)
LOWFIELD_SPACING = (1.6, 1.6, 5.0)  # mm
NOISE_FRACTION = 0.06               # noise std as a fraction of the 99th-percentile intensity
TRUE_ROTATION_DEG = (6.0, -4.0, 9.0)  # head movement around x, y, z (degrees)
TRUE_SHIFT_MM = (7.0, -12.0, 5.0)     # head movement along x, y, z (mm)
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


def mask_background(t1, inv2):
    """MP2RAGE T1w images have loud noise outside the head. Zero it using the INV2 image."""
    inv2 = sitk.Cast(inv2, sitk.sitkFloat32)
    head = sitk.OtsuThreshold(inv2, 0, 1)
    head = sitk.BinaryMorphologicalClosing(head, [3, 3, 3])
    head = sitk.BinaryFillhole(head)
    return sitk.Cast(t1, sitk.sitkFloat32) * sitk.Cast(head, sitk.sitkFloat32)


def simulate_low_field(flair, true_tf, rng):
    """Resample the FLAIR onto a coarse grid while applying the known head movement, then add noise."""
    size = [int(round(n * s / ls)) for n, s, ls in
            zip(flair.GetSize(), flair.GetSpacing(), LOWFIELD_SPACING)]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(LOWFIELD_SPACING)
    grid.SetOrigin(flair.GetOrigin())
    grid.SetDirection(flair.GetDirection())
    # Smooth first so the coarse voxels average their area instead of aliasing.
    # (Only in-plane: the FLAIR slices are already 5 mm thick.)
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


def register(fixed, moving, init, shrink=(4, 2, 1), smooth_mm=(2, 1, 0)):
    """Rigid (rotation + translation) registration that maximizes mutual information."""
    reg = sitk.ImageRegistrationMethod()
    # Mutual information works across contrasts (T1 vs FLAIR): it rewards a consistent
    # mapping between the two intensity patterns, not equal intensities.
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.NONE)  # use every voxel; random subsets made it jitter
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=400,
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
    print(f"Optimizer stopped: {reg.GetOptimizerStopConditionDescription()}")
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


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    t1 = mask_background(sitk.ReadImage(str(T1_PATH)), sitk.ReadImage(str(INV2_PATH)))
    flair = sitk.ReadImage(str(FLAIR_PATH))
    moving = normalize(t1)

    # Step A: measure the small same-visit offset between T1 and the full-quality FLAIR.
    print("Step A: T1 -> full-quality FLAIR (finds the small same-visit offset)")
    no_move = euler((0, 0, 0), (0, 0, 0), image_center(flair))
    visit_tf, _ = register(normalize(sitk.Cast(flair, sitk.sitkFloat32)), moving, no_move,
                           shrink=(2, 1), smooth_mm=(1, 0))

    # Step B: build the simulated low-field scan with a known head movement.
    move_tf = euler(TRUE_ROTATION_DEG, TRUE_SHIFT_MM, image_center(flair))
    low = simulate_low_field(flair, move_tf, rng)
    fixed = normalize(low)
    sitk.WriteImage(low, str(OUT / "simulated_lowfield.nii.gz"))
    # A low-field point x sits at move_tf(x) in FLAIR space, then visit_tf maps it into T1 space.
    true_tf = sitk.CompositeTransform([visit_tf, move_tf])  # applied right to left

    # Step C: the real test. Start from the scanner coordinates (no guess about the movement).
    print("Step C: T1 -> simulated low-field scan")
    identity = sitk.Euler3DTransform()
    start = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    est_tf, history = register(fixed, moving, start)

    before_mi = mutual_information(fixed, moving, identity)
    after_mi = mutual_information(fixed, moving, est_tf)
    before_err = registration_error_mm(fixed, identity, true_tf)
    after_err = registration_error_mm(fixed, est_tf, true_tf)

    aligned = sitk.Resample(t1, low, est_tf, sitk.sitkLinear, 0.0)
    sitk.WriteImage(aligned, str(OUT / "t1_aligned_to_lowfield.nii.gz"))
    sitk.WriteTransform(est_tf, str(OUT / "t1_to_lowfield.tfm"))

    results = {
        "mutual_information": {"before": round(before_mi, 4), "after": round(after_mi, 4)},
        "mean_error_mm": {"before": round(before_err[0], 2), "after": round(after_err[0], 2)},
        "max_error_mm": {"before": round(before_err[1], 2), "after": round(after_err[1], 2)},
        "true_rotation_deg": TRUE_ROTATION_DEG,
        "true_shift_mm_about_center": TRUE_SHIFT_MM,
        "iterations": len(history),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    np.savetxt(OUT / "metric_history.txt", history)
    print(json.dumps(results, indent=2))

    from figures import save_figures
    save_figures(fixed, moving, identity, est_tf, history, results, OUT)


if __name__ == "__main__":
    main()
