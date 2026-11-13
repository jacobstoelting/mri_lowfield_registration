"""normalize, resample_spacing and mask_background: intensities and image geometry."""
import numpy as np
import pytest
import SimpleITK as sitk

from register import mask_background, normalize, resample_spacing


def same_geometry(a, b, atol=1e-9):
    return (np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol))


def corner_mm(img):
    """Physical position of the outer corner of the volume (edge of the first voxel, not its center)."""
    return np.array(img.TransformContinuousIndexToPhysicalPoint([-0.5] * 3))


def far_corner_mm(img):
    return np.array(img.TransformContinuousIndexToPhysicalPoint([n - 0.5 for n in img.GetSize()]))


# --- normalize -----------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["phantom", "oblique"])
def test_normalize_range_and_geometry(which, phantom, oblique_image):
    img = mask_background(phantom.image) if which == "phantom" else oblique_image
    out = normalize(img)
    a = sitk.GetArrayViewFromImage(out)

    assert a.min() >= 0.0 and a.max() <= 1.0
    assert a.max() == 1.0  # the top 1% is clipped to exactly 1
    assert out.GetSize() == img.GetSize()
    assert same_geometry(out, img)


# --- resample_spacing ----------------------------------------------------------------------

@pytest.mark.parametrize("spacing", [2.0, (1.6, 1.6, 5.0), (0.7, 1.3, 1.0)])
def test_resample_spacing_geometry(oblique_image, spacing):
    img = oblique_image
    out = resample_spacing(img, spacing)
    target = np.broadcast_to(np.asarray(spacing, dtype=float), (3,))

    assert np.allclose(out.GetSpacing(), target)
    assert np.allclose(out.GetDirection(), img.GetDirection())
    # The grid starts at the same physical corner (exactly, by construction) ...
    assert np.allclose(corner_mm(out), corner_mm(img), atol=1e-6)
    # ... and ends within half a new voxel of the old far corner (the size is rounded to whole voxels).
    d = np.array(img.GetDirection()).reshape(3, 3)
    gap_per_axis = np.abs(d.T @ (far_corner_mm(out) - far_corner_mm(img)))
    assert np.all(gap_per_axis <= target / 2 + 1e-6), gap_per_axis


def test_resample_spacing_keeps_anatomy_in_place(phantom):
    """Resampling must not shift the image content: the bright-blob centroid stays put."""
    img = phantom.image
    out = resample_spacing(img, (3.0, 3.0, 6.0))

    def bright_centroid_mm(im):
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(im > 0.8, sitk.sitkUInt8))
        return np.array(stats.GetCentroid(1))

    assert np.linalg.norm(bright_centroid_mm(out) - bright_centroid_mm(img)) < 1.5


# --- mask_background -----------------------------------------------------------------------

def shrink(mask, voxels):
    """Erode (voxels > 0) or dilate (voxels < 0) a bool (z, y, x) array."""
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    f = sitk.BinaryErode if voxels > 0 else sitk.BinaryDilate
    return sitk.GetArrayFromImage(f(img, [abs(voxels)] * 3)).astype(bool)


def test_mask_background_zeros_air_keeps_head(phantom):
    raw = sitk.GetArrayFromImage(phantom.image)
    out = sitk.GetArrayFromImage(mask_background(phantom.image))

    # Air: everything a few voxels outside the head is set to exactly 0 (it was noisy before).
    air = ~shrink(phantom.head, -3)
    assert raw[air].max() > 0  # there really was noise to remove
    assert np.all(out[air] == 0)

    # Inside: everything a few voxels inside the head edge is kept unchanged ...
    inside = shrink(phantom.head, 3)
    np.testing.assert_array_equal(out[inside], raw[inside])
    # ... including the dark CSF region, which is darker than the Otsu threshold.
    assert raw[phantom.csf].mean() < 0.2
    np.testing.assert_array_equal(out[phantom.csf], raw[phantom.csf])
