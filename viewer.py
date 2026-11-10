"""3D viewer (napari) of the simulated low-field scan vs the 3T T1, before and after alignment.

Run from the repo root after `python register.py`:
    conda activate mri
    python viewer.py

All three volumes live on the low-field voxel grid, so napari can overlay them directly.
Note: napari draws plain array axes and ignores the image direction matrix, so the head
may look tilted/flipped compared with 3D Slicer. The layers still line up with each other.
"""
from pathlib import Path

import napari
import SimpleITK as sitk

from register import INV2_PATH, OUT, T1_PATH, mask_background, normalize

LOWFIELD_PATH = OUT / "simulated_lowfield.nii.gz"
ALIGNED_PATH = OUT / "t1_aligned_to_lowfield.nii.gz"


def load_volumes():
    """Return (low-field, T1 before, T1 after) as normalized SimpleITK images on the low-field grid."""
    for p in (LOWFIELD_PATH, ALIGNED_PATH):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p}. Run `python register.py` first.")

    low = sitk.ReadImage(str(LOWFIELD_PATH), sitk.sitkFloat32)
    after = sitk.ReadImage(str(ALIGNED_PATH), sitk.sitkFloat32)

    # "Before": the masked T1 placed on the low-field grid using only scanner coordinates
    # (identity transform), i.e. ignoring the head movement. Same masking as register.py.
    t1 = mask_background(sitk.ReadImage(str(T1_PATH)), sitk.ReadImage(str(INV2_PATH)))
    # Background = 0 so normalize() (which looks at voxels > 0) only sees the head.
    before = sitk.Resample(t1, low, sitk.Transform(), sitk.sitkLinear, 0.0)

    return normalize(low), normalize(before), normalize(after)


def main():
    low, before, after = load_volumes()
    arrays = {name: sitk.GetArrayFromImage(img)  # [z, y, x]
              for name, img in (("low-field", low), ("T1 before", before), ("T1 after", after))}
    for name, a in arrays.items():
        print(f"{name:10s} shape {a.shape}")
    assert len({a.shape for a in arrays.values()}) == 1, "volumes are not on the same grid"

    scale = tuple(reversed(low.GetSpacing()))  # spacing is (x, y, z); arrays are [z, y, x]
    print(f"scale (z, y, x) mm: {scale}")

    viewer = napari.Viewer(title="Low-field vs 3T T1")
    viewer.add_image(arrays["low-field"], name="low-field", scale=scale,
                     colormap="green", blending="additive", contrast_limits=(0, 1))
    viewer.add_image(arrays["T1 before"], name="T1 before", scale=scale,
                     colormap="magenta", blending="additive", contrast_limits=(0, 1), visible=False)
    viewer.add_image(arrays["T1 after"], name="T1 after", scale=scale,
                     colormap="magenta", blending="additive", contrast_limits=(0, 1))
    napari.run()


if __name__ == "__main__":
    main()
