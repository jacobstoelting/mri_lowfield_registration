# Aligning real 64 mT portable MRI to 3T brain scans

**[Open the 3D viewer](https://jacobstoelting.github.io/mri_lowfield_registration/)** (runs in the browser, nothing to install)

This project takes brain scans of the same person from two very different MRI scanners and
lines them up automatically:

- a **portable 64 mT low-field scanner** (Hyperfine Swoop): coarse 1.6 x 1.6 x 5 mm voxels,
  made at the bedside
- a **3T hospital scanner** (Philips Achieva): sharp 1 mm voxels

Then it measures how well they line up, and shows the result in an interactive 3D viewer.
It uses Python and SimpleITK, with a NiiVue page for the viewer.

![Before and after overlay](results/overlay_before_after.png)

## Why line them up?

Portable low-field MRI can scan patients where a 3T scanner can't go, for example in the
ICU, but its images are much coarser. Registration finds the rotation and shift that map
each point in one scan to the same piece of anatomy in the other. That step comes before
anything that compares or combines the scans:

- **Follow-up.** Compare a new bedside scan with an earlier 3T scan, location by location
  ("have the ventricles grown?").
- **Carrying 3T information over.** Outlines drawn on the sharp 3T scan (brain regions,
  lesions, a surgical plan) can be placed on the low-field scan.
- **Low-field enhancement.** Training or testing a model that makes 64 mT images look like
  3T needs pairs where each voxel matches the same anatomy.

This project does the alignment. It does not upscale or change the low-field images.

## Results

All 11 volunteers scanned on both machines. Fixed image: 64 mT T2 (session 1). Moving
image: 3T T1.

| Subject | Mutual information before → after | Self-consistency error, mean | Self-consistency error, worst |
|---|---|---|---|
| sub-0011 | 0.148 → 0.401 | 0.57 mm | 1.11 mm |
| sub-0015 | 0.228 → 0.418 | 0.82 mm | 1.64 mm |
| sub-0023 | 0.339 → 0.422 | 0.67 mm | 1.20 mm |
| sub-0025 | 0.217 → 0.361 | 0.71 mm | 1.45 mm |
| sub-0027 | 0.305 → 0.420 | 1.30 mm | 2.11 mm |
| sub-0035 | 0.292 → 0.435 | 0.65 mm | 1.01 mm |
| sub-0046 | 0.316 → 0.429 | 0.14 mm | 0.27 mm |
| sub-0047 | 0.265 → 0.444 | 0.61 mm | 1.06 mm |
| sub-0048 | 0.224 → 0.402 | 0.15 mm | 0.20 mm |
| sub-0064 | 0.198 → 0.390 | 0.84 mm | 1.64 mm |
| sub-0066 | 0.176 → 0.384 | 0.29 mm | 0.61 mm |
| **Median** | | **0.65 mm** | |

Every subject is under 1.3 mm on average, smaller than one 64 mT voxel (1.6 x 1.6 x 5 mm).
`sub-0035` is the one shown in the figure above; the viewer can show any of the 11.
Results vary by a few hundredths of a millimeter between runs for `sub-0011`, whose runs
use the full iteration budget, so multithreaded rounding differences can nudge where they stop.

- **Before** means the two image centers lined up, with no rotation. In raw scanner
  coordinates the scans don't overlap at all, because the two scanners' coordinate systems
  are about 22 cm apart (mutual information 0.00).
- **Self-consistency:** real scans have no known correct answer, so the 3T scan is moved
  by a known rotation and shift (6, -4, 9 degrees; 7, -12, 5 mm) and registered again from
  scratch. The new answer should equal the first one plus that known movement. The error
  is how far apart the two answers put each point of the head.
- **Two starting guesses** (image centers, and centers of mass) are registered for every
  subject, keeping the one with higher mutual information. For 3 subjects they end in
  different places, but the loser always scores clearly lower (for example 0.320 vs 0.401
  for `sub-0011`), so the choice is clear. Starting from image centers alone would have
  given a wrong answer, about 20 mm off, for `sub-0011` and `sub-0066`.

Full results: `results/all_subjects.csv`. Contact sheet with the 3T outlines on each 64 mT
scan: `results/all_subjects_overlay.png`.

### What went wrong first

The first run over all 11 subjects looked worse: 3 subjects had self-consistency errors of
3.7-6.1 mm. A diagnosis showed the original registration, not the re-run, was the weak one:
mapped back, the re-run's answer had *higher* mutual information. Every one of those runs
had stopped at the optimizer's limit of 400 iterations per resolution level, while every
good subject stopped because its step had shrunk to nothing (converged). The optimizer had
been cut off while still improving. Raising the limit to 2000 brought those subjects to
0.14-0.61 mm and left the already-converged subjects unchanged. Runs that converge stop
early anyway, so the higher limit only costs time where it's needed; the slowest subject
takes about 2 minutes including the self-consistency check.

## The 3D viewer

[jacobstoelting.github.io/mri_lowfield_registration](https://jacobstoelting.github.io/mri_lowfield_registration/)
(built on [NiiVue](https://github.com/niivue/niivue), source in `docs/`)

- **Subject menu.** Any of the 11 volunteers (with their self-consistency error). The
  choice, and the before/after and view settings, stay in the address bar, so a copied
  link reopens the same view.
- **Side by side, linked.** The 64 mT scan on the left and the 3T scan on the right, both in
  grayscale. Clicking, scrolling, zooming or rotating in one view does the same in the
  other, so the crosshair always points at the same anatomy in both. That is what the
  registration makes possible.
- **3T outlines (yellow)** on the 64 mT scan show the alignment. They are edges of the 3T
  scan (3D Canny), and edges work across contrasts even though T1 and T2 show tissue almost
  oppositely (fluid is dark on T1 and bright on T2). In **After** they trace the 64 mT
  ventricles, brain and skull; in **Before** they visibly miss. **Flicker** alternates the
  two. Toggle the outlines with the button or the O key. They start off in "3 planes + 3D",
  where they would also cover the 64 mT 3D render.
- **The blockiness is real.** The 64 mT scan is shown exactly as recorded: 5 mm slices,
  which look like stair-steps in the coronal and sagittal views. It is moved into place by
  rewriting its file header (position and orientation), not by resampling, so nothing is
  smoothed or invented. Both views draw on the 1 mm 3T grid, so the 3T stays sharp.

## How it works (`register.py`)

1. **Load.** The 3T T1 is put on a 1 mm isotropic grid. That's its real measured
   resolution (the file is interpolated to 0.49 mm in-plane).
2. **Clean.** Zero the background with a head mask: Otsu threshold on a smoothed copy, then
   fill holes. Clip each scan to its 1st-99th percentile and scale to 0-1.
3. **Starting guess.** Line up the image centers (GEOMETRY), and separately the centers of
   mass (MOMENTS). Register from both and keep the higher mutual information. Here both
   reach the same answer.
4. **Rigid registration** with `sitk.ImageRegistrationMethod`:
   - `Euler3DTransform`: 3 rotations + 3 shifts, since it's the same head.
   - Mattes mutual information. It works across contrasts because it only asks that one
     image's intensities predict the other's, not that they're equal.
   - Regular-step gradient descent, coarse-to-fine (images shrunk 4x, 2x, then full size),
     up to 2000 iterations per level. It stops early once its step shrinks below a minimum.
5. **Self-consistency check** (above), then write the outputs and figures.

Notes:

- **Bias field.** Brightness in the 64 mT T2 varies by about +-15% across the head.
  Correcting it with N4 changed the result by only 0.2 mm on average, so it isn't used.
- **Orientation.** NIfTI stores positions in RAS, while SimpleITK works in LPS (x and y
  flipped). SimpleITK converts on read, and each image's direction matrix records how its
  voxel grid is tilted, so scans from both scanners share one coordinate space without
  manual flipping.

## Data

[Paired 64mT and 3T Brain MRI Scans of Healthy Subjects for Neuroimaging Research](https://doi.org/10.5281/zenodo.20281403)
(Leiden University Medical Center), Zenodo record 20281403 (v5), **MIT license**.

- 86 low-field sessions from 66 healthy volunteers. 11 of the volunteers were also scanned
  at 3T, with the same subject IDs in both folders.
- BIDS format (NIfTI), faces removed.

> van den Broek, R.B., Lena, B., & Webb, A. (2026). Paired 64mT and 3T Brain MRI Scans of
> Healthy Subjects for Neuroimaging Research (v5). Zenodo. https://doi.org/10.5281/zenodo.20281403

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py      # 1.4 GB from Zenodo, unpacked to data/leiden/
python register.py           # one subject (sub-0035), 20-30 seconds; writes results/
python smoke_test.py         # optional: checks sub-0035 still registers (MI up, error under 1.5 mm)
python all_subjects.py       # all 11 paired subjects, about 9 minutes
python export_viewer.py      # writes every subject's viewer files to docs/data/ (needs all_subjects.py first)
python -m http.server -d docs 8000   # optional: preview the viewer at http://localhost:8000
```

The subject and which 64 mT scan to use (`T2w`, `FLAIR` or `T1w`) are constants at the
top of `register.py`.

## Files

| File | What it does |
|---|---|
| `download_data.py` | Downloads and unpacks the Zenodo dataset into `data/` |
| `register.py` | Loads, cleans and registers the scans; self-consistency test; writes `results/` |
| `all_subjects.py` | Runs `register.py`'s pipeline for all 11 paired subjects; summary table, contact sheet, and each subject's transform (`results/transforms/`) |
| `smoke_test.py` | Quick check after code changes: registers sub-0035 and fails if the result got worse |
| `figures.py` | Before/after overlay, checkerboard and optimizer-progress figures |
| `export_viewer.py` | Writes each subject's scans and 3T outlines to `docs/data/sub-XXXX/`, plus the subject list |
| `docs/index.html` | The NiiVue viewer (served by GitHub Pages) |
| `results/` | Figures, `results.json`, and the transform `t1_to_lowfield.tfm` |

`register.py` also keeps `simulate_low_field`, which makes a fake low-field scan with a
known movement. It's unused in the main pipeline and is there for a planned robustness test.

## Limits and next steps

- **11 subjects, one contrast.** All paired volunteers are covered, but only with the 64 mT
  T2 from their first session. The 64 mT T1 and FLAIR, and repeat sessions, are untested.
- **Rigid only.** It doesn't model low-field geometric distortion. Also not sure if significant
  brain changes would alter results.
- **Self-consistency is not ground truth.** It shows the method lands in the same place
  again, not that the place is anatomically exact, and it is optimistic because the moved
  copy is the same 3T image. Manually placed landmarks would be a stronger check.
