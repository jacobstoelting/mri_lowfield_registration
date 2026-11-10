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

Subject `sub-0035`. Fixed image: 64 mT T2. Moving image: 3T T1.

| | Before | After |
|---|---|---|
| Mutual information (higher = better aligned) | 0.292 | 0.435 |
| Self-consistency error, mean over the head | 18.4 mm | **0.65 mm** |
| Self-consistency error, worst case | 27.8 mm | 1.02 mm |

- **Before** means the two image centers lined up, with no rotation. In raw scanner
  coordinates the scans don't overlap at all, because the two scanners' coordinate systems
  are about 22 cm apart (mutual information 0.00).
- **Self-consistency:** real scans have no known correct answer, so the 3T scan is moved
  by a known rotation and shift (6, -4, 9 degrees; 7, -12, 5 mm) and registered again from
  scratch. The new answer should equal the first one plus that known movement. It does, to
  within 0.65 mm on average, less than half a 64 mT voxel.

## The 3D viewer

[jacobstoelting.github.io/mri_lowfield_registration](https://jacobstoelting.github.io/mri_lowfield_registration/)
(built on [NiiVue](https://github.com/niivue/niivue), source in `docs/`)

- **Side by side, linked.** The 64 mT scan on the left and the 3T scan on the right, both in
  grayscale. Clicking, scrolling, zooming or rotating in one view does the same in the
  other, so the crosshair always points at the same anatomy in both. That is what the
  registration makes possible.
- **3T outlines (yellow)** on the 64 mT scan show the alignment. They are edges of the 3T
  scan (3D Canny), and edges work across contrasts even though T1 and T2 show tissue almost
  oppositely (fluid is dark on T1 and bright on T2). In **After** they trace the 64 mT
  ventricles, brain and skull; in **Before** they visibly miss. **Flicker** alternates the
  two. Toggle the outlines with the button or the O key.
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
   - Regular-step gradient descent, coarse-to-fine (images shrunk 4x, 2x, then full size).
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
python register.py           # about 30 seconds; writes results/
python export_viewer.py      # writes the viewer's scans to docs/data/
python -m http.server -d docs 8000   # optional: preview the viewer at http://localhost:8000
```

The subject and which 64 mT scan to use (`T2w`, `FLAIR` or `T1w`) are constants at the
top of `register.py`.

## Files

| File | What it does |
|---|---|
| `download_data.py` | Downloads and unpacks the Zenodo dataset into `data/` |
| `register.py` | Loads, cleans and registers the scans; self-consistency test; writes `results/` |
| `figures.py` | Before/after overlay, checkerboard and optimizer-progress figures |
| `export_viewer.py` | Writes the scans and 3T outlines the viewer loads to `docs/data/` |
| `docs/index.html` | The NiiVue viewer (served by GitHub Pages) |
| `results/` | Figures, `results.json`, and the transform `t1_to_lowfield.tfm` |

`register.py` also keeps `simulate_low_field`, which makes a fake low-field scan with a
known movement. It's unused in the main pipeline and is there for a planned robustness test.

## Limits and next steps

- **One subject, one contrast.** 10 more paired subjects are in the dataset, plus 64 mT T1
  and FLAIR scans. Running them all would show how reliable the method is.
- **Rigid only.** It doesn't model low-field geometric distortion or brain changes between
  the two visits, which were on different days.
- **Self-consistency is not ground truth.** It shows the method lands in the same place
  again, not that the place is anatomically exact, and it is optimistic because the moved
  copy is the same 3T image. Manually placed landmarks would be a stronger check.
