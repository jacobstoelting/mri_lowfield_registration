"""Export the volumes for the NiiVue web viewer (docs/index.html).

Run from the repo root after `python register.py`:
    conda activate mri
    python export_viewer.py
    python -m http.server -d docs 8000     # then open http://localhost:8000

Writes to docs/data/ (all on the low-field grid, scaled 0-255 as uint8 so they load fast):
    lowfield.nii.gz   simulated low-field scan (noise outside the head zeroed for display)
    t1_before.nii.gz  3T T1 placed using scanner coordinates only (no correction)
    t1_after.nii.gz   3T T1 after the rigid registration
    results.json      copy of results/results.json (shown on the page)

The files keep the scanner orientation (origin + direction), so NiiVue shows the head
in correct anatomical orientation.
"""
import shutil
from pathlib import Path

import SimpleITK as sitk

from register import INV2_PATH, OUT, T1_PATH, mask_background, normalize

LOWFIELD_PATH = OUT / "simulated_lowfield.nii.gz"
ALIGNED_PATH = OUT / "t1_aligned_to_lowfield.nii.gz"
DOCS_DATA = Path("docs") / "data"


def to_uint8(img):
    """normalize() gives 0-1 floats; store as 0-255 bytes (4x smaller, plenty for display)."""
    return sitk.Cast(sitk.Round(normalize(img) * 255.0), sitk.sitkUInt8)


def main():
    for p in (LOWFIELD_PATH, ALIGNED_PATH):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run `python register.py` first.")
    DOCS_DATA.mkdir(parents=True, exist_ok=True)

    low = sitk.ReadImage(str(LOWFIELD_PATH), sitk.sitkFloat32)
    # Display only: zero the noise outside the head (mask from a smoothed copy of the scan
    # itself), otherwise the 3D render is a solid box of noise. Noise inside the head stays.
    low = mask_background(low, sitk.SmoothingRecursiveGaussian(low, 2.0))
    after = sitk.ReadImage(str(ALIGNED_PATH), sitk.sitkFloat32)
    # "Before": masked T1 on the low-field grid with an identity transform, i.e. trusting the
    # scanner coordinates and ignoring the head movement. Background 0 so normalize() skips it.
    t1 = mask_background(sitk.ReadImage(str(T1_PATH)), sitk.ReadImage(str(INV2_PATH)))
    before = sitk.Resample(t1, low, sitk.Transform(), sitk.sitkLinear, 0.0)

    for name, img in (("lowfield", low), ("t1_before", before), ("t1_after", after)):
        out = DOCS_DATA / f"{name}.nii.gz"
        sitk.WriteImage(to_uint8(img), str(out), useCompression=True)
        print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")

    shutil.copy(OUT / "results.json", DOCS_DATA / "results.json")
    print(f"wrote {DOCS_DATA / 'results.json'}")


if __name__ == "__main__":
    main()
