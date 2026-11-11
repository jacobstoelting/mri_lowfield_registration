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

**Median error 1.15 mm by the stricter check (consistency loop), 0.67 mm by self-consistency.
Both are smaller than one 64 mT voxel (1.6 x 1.6 x 5 mm).**

| Subject | Mutual information before → after | Self-consistency error, mean | Loop error, mean | Loop error, worst |
|---|---|---|---|---|
| sub-0011 | 0.148 → 0.401 | 0.67 mm | 2.67 mm | 4.90 mm |
| sub-0015 | 0.228 → 0.418 | 0.82 mm | 0.58 mm | 0.99 mm |
| sub-0023 | 0.339 → 0.422 | 0.67 mm | 1.09 mm | 1.89 mm |
| sub-0025 | 0.217 → 0.361 | 0.71 mm | 1.24 mm | 2.13 mm |
| sub-0027 | 0.304 → 0.420 | 1.29 mm | 1.15 mm | 1.62 mm |
| sub-0035 | 0.291 → 0.435 | 0.65 mm | 1.44 mm | 2.50 mm |
| sub-0046 | 0.316 → 0.428 | 0.14 mm | 0.72 mm | 1.31 mm |
| sub-0047 | 0.265 → 0.444 | 0.60 mm | 0.86 mm | 1.80 mm |
| sub-0048 | 0.224 → 0.402 | 0.15 mm | 0.61 mm | 1.22 mm |
| sub-0064 | 0.198 → 0.390 | 0.84 mm | 3.31 mm | 5.60 mm |
| sub-0066 | 0.176 → 0.384 | 0.30 mm | 1.78 mm | 3.09 mm |
| **Median** | | **0.67 mm** | **1.15 mm** | |

`sub-0035` is the one shown in the figure above; the viewer can show any of the 11. Every
registration converged (the optimizer stopped because its steps had shrunk to nothing, not
because it ran out of iterations).

- **Before** means the two image centers lined up, with no rotation. In raw scanner
  coordinates the scans don't overlap at all, because the two scanners' coordinate systems
  are about 22 cm apart (mutual information 0.00).
- **Self-consistency:** real scans have no known correct answer, so the 3T scan is moved
  by a known rotation and shift (6, -4, 9 degrees; 7, -12, 5 mm) and registered again from
  scratch. The new answer should equal the first one plus that known movement. The error
  is how far apart the two answers put each point of the head. This is optimistic: both
  runs register the same 3T image, so it mostly tests that the optimizer lands in the same
  place again.
- **Consistency loop (stricter):** the 64 mT scan is also registered to a *different* 3T
  image, the T2 from the same 3T session, and that T2 is registered to the 3T T1. The two
  routes from the 64 mT scan to the 3T T1 (directly, and through the 3T T2) should send
  every point of the head to the same place; the loop error is how far apart they put it.
  The step between the two 3T scans also shows the head moved 0.6-3.0 mm between those two
  3T sequences, so assuming it stayed still would have been wrong.
- **`sub-0064` (3.3 mm loop error)** is a real disagreement, not a failed run: every step
  converged and both starting guesses agree. The registrations to the 3T T1 and to the 3T T2
  settle in slightly different places. The loop can show that they disagree but not which
  one is wrong. The 3T T2 has 3 mm slices and covers only 15 cm of the head, so it gives the
  registration less to hold on to.
- **Two starting guesses** (image centers, and centers of mass) are registered for every
  subject, keeping the one with higher mutual information. Now that every run converges,
  both end in the same place for every subject (within 0.01 mm).

Full results: `results/all_subjects.csv` and `results/consistency_loop.json`. Contact sheet
with the 3T outlines on each 64 mT scan: `results/all_subjects_overlay.png`.

### Repeat 64 mT sessions

Five of the volunteers had the 64 mT scan repeated on other days (9 repeat sessions). Their
heads sat 3-20 mm and 1-10 degrees away from the first session's position in the scanner,
so scans from different days have to be registered before they can be compared. A second
consistency loop goes through a different day: 64 mT session 1 → 64 mT session N → 3T T1,
against session 1 → 3T T1 directly. The loop error has a median of **0.89 mm** (worst
1.15 mm). One registration, `sub-0027` session 5 to the 3T T1, did not converge: after 20000
iterations its mutual information was still slowly rising. Faster optimizers converged
quickly but onto wrong answers 4-10 mm away, so this pair has many shallow false optima; the
slow climb still ends 0.77 mm from the route through session 1. Full table:
`results/repeat_sessions.json`.

### What went wrong first

The first run over all 11 subjects looked worse: 3 subjects had self-consistency errors of
3.7-6.1 mm. A diagnosis showed the original registration, not the re-run, was the weak one:
mapped back, the re-run's answer had *higher* mutual information. Every one of those runs
had stopped at the optimizer's limit of 400 iterations per resolution level, while every
good subject stopped because its step had shrunk to nothing (converged). The optimizer had
been cut off while still improving. Raising the limit to 2000 brought those subjects to
0.14-0.61 mm and left the already-converged subjects unchanged.

The consistency loop then exposed that 2000 still wasn't enough for `sub-0011`: all three
resolution levels hit the limit, with mutual information still slowly rising. Its two
starting guesses ended 23 mm apart (the worse one was simply cut off further from the
answer), and repeat runs differed by 0.65 mm. With a limit of 20000 its full-resolution
level converges after under 1000 iterations, both starts agree, and its answer moved 0.2 mm.
The other subjects changed by 0.06 mm or less. Each run now records whether it converged,
and an unconverged run is flagged. Runs that converge stop early anyway, so the higher
limit only costs time where it's needed; the slowest subject (`sub-0066`) takes about 10
minutes including the self-consistency check.

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
     up to 20000 iterations per level. It stops early once its step shrinks below a minimum
     (converged); every run records whether it did.
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
python all_subjects.py       # all 11 paired subjects, about 25 minutes
python consistency_loop.py   # stricter accuracy check through the 3T T2, about 10 minutes
python repeat_sessions.py    # repeat 64 mT sessions: head position and a second loop, about 25 minutes
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
| `consistency_loop.py` | Consistency loop: registers each 64 mT T2 to the 3T T2 too, and the 3T T2 to the 3T T1, and checks the two routes agree |
| `repeat_sessions.py` | Registers each repeat 64 mT session to session 1 and checks the loop through it to the 3T T1 |
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
- **Neither check is ground truth.** Self-consistency shows the method lands in the same
  place again, and is optimistic because the moved copy is the same 3T image. The
  consistency loop uses a different 3T image, but both of its routes start from the same
  64 mT scan, so a distortion in that scan would bend both the same way and stay hidden.
  Manually placed landmarks would be a stronger check.
