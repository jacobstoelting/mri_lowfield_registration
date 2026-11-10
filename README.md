# Aligning a 3T brain MRI to a low-field scan

A small Python + SimpleITK project. It takes a high-quality 3T brain scan and lines it up
with a lower-quality "low-field" scan of the same person, then measures how well it worked.

Portable low-field MRI scanners (around 64 mT) make blurrier, noisier images than hospital
3T scanners. If a patient already has an older 3T scan, aligning it to the new low-field
scan lets you compare the two or use the sharp scan to help read the blurry one. The
first step is registration: finding the rotation and shift that line the scans up.

![Before and after overlay](results/overlay_before_after.png)

## Results

| | Before | After |
|---|---|---|
| Mutual information (higher = more aligned) | 0.347 | 0.536 |
| Mean error over the head | 16.0 mm | 0.31 mm |
| Worst-case error over the head | 29.1 mm | 0.44 mm |

The low-field voxels are 1.6 x 1.6 x 5 mm, so the remaining error is smaller than one voxel.

## How it works

1. **Data.** One subject from the [MPI-Leipzig Mind-Brain-Body dataset](https://openneuro.org/datasets/ds000221)
   (OpenNeuro ds000221, CC0): a 3T T1-weighted scan (1 mm voxels) and a FLAIR scan
   (0.43 x 0.43 x 5 mm, tilted slices) from the same visit.
2. **Clean the T1.** The MP2RAGE T1 has strong noise outside the head, so it is masked
   using the second inversion image.
3. **Simulate a low-field scan** from the FLAIR: smooth and resample to 1.6 x 1.6 x 5 mm
   voxels, add Rician noise (the kind of noise MRI magnitude images have), and move the
   head by a known amount (6, -4 and 9 degrees of rotation, 7, -12 and 5 mm of shift).
4. **Normalize intensities.** Both scans are clipped to their 1st-99th percentile and
   scaled to 0-1, so a few very bright voxels don't control the result.
5. **Rigid registration** (`sitk.ImageRegistrationMethod`):
   - Transform: `Euler3DTransform`, 3 rotations + 3 shifts, since it's the same head.
   - Metric: Mattes mutual information. The two scans have different contrasts (in T1,
     fluid is dark and white matter bright; FLAIR differs), so matching intensities
     directly would fail. Mutual information only asks that one image's intensities
     predict the other's.
   - Optimizer: regular-step gradient descent, run coarse-to-fine (images shrunk 4x, then
     2x, then full size) so it doesn't get stuck on small details early.
6. **Check the answer.** The T1 and FLAIR were taken minutes apart, so first the T1 is
   registered to the full-quality FLAIR to measure the small offset from the person moving
   during the visit (about 1 degree and 1.6 mm). The correct answer for the low-field scan
   is that offset plus the known movement. The error is the distance between where the
   estimated transform and the correct one send each head voxel.

### Orientation (LPS vs RAS)

NIfTI files describe positions in RAS (+x toward the patient's Right, +y Anterior,
+z Superior). ITK/SimpleITK and DICOM use LPS (+x toward Left, +y Posterior, +z Superior),
so SimpleITK flips the sign of x and y when it reads a NIfTI. Every image here then lives
in LPS millimeters, and each image's direction matrix says how its voxel grid is tilted in
that space. That is why the tilted FLAIR slices and the straight T1 grid can be compared
without any manual flipping. (Mixing the two conventions by hand is a classic way to get
a left-right flipped result.)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py   # about 30 MB from OpenNeuro
python register.py        # about 30 seconds
```

Outputs go to `results/`: the overlay and checkerboard images, the optimizer curve,
`results.json`, the transform (`t1_to_lowfield.tfm`), and the aligned T1 as NIfTI.

## Limits

- The low-field scan is simulated from a 3T FLAIR, not taken on a real low-field scanner.
  Real 64 mT images also have different contrast and distortions.
- The "correct answer" includes a same-visit offset found by the same method on full-quality
  images, so the error measures how much the low-field degradation hurts, not absolute accuracy.
- One subject and one movement.
