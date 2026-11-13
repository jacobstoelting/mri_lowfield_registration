"""Shared fixtures: a small synthetic 3D "head" phantom, so the tests need no real scans.

The phantom is a 64 x 64 x 64 volume with 2 mm voxels (12.8 cm across). It has:
  - an ellipsoid "head" of medium brightness,
  - a bright blob, a medium-dark blob and a dark CSF-like blob, all placed off-center and
    asymmetrically, so a rotation or shift of the head changes the image in a detectable way,
  - a little Rician-style noise everywhere (also in the air around the head), like real MRI.
"""
from dataclasses import dataclass

import numpy as np
import pytest
import SimpleITK as sitk

SIZE = 64
SPACING_MM = 2.0
ORIGIN_MM = (-63.0, -70.0, -55.0)  # arbitrary scanner position, not zero on purpose

HEAD_CENTER_MM = (0.0, -6.0, 9.0)          # head center (x, y, z), mm
HEAD_SEMI_AXES_MM = (38.0, 46.0, 34.0)     # smaller than the box so moved copies stay inside

# (center mm, radius mm, intensity); head tissue is 0.6
BLOBS = [
    ((14.0, 8.0, 18.0), 9.0, 1.0),    # bright, right-front-top
    ((-16.0, -22.0, 2.0), 7.0, 0.3),  # darker, left-back
]
CSF_CENTER_MM = (-6.0, 10.0, 6.0)  # dark "ventricle", well inside the head
CSF_RADII_MM = (6.0, 12.0, 5.0)
TISSUE, CSF = 0.6, 0.05
NOISE_SD = 0.03


@dataclass
class Phantom:
    image: sitk.Image        # noisy phantom, background not masked (like a raw scan)
    head: np.ndarray         # bool (z, y, x): inside the ellipsoid
    csf: np.ndarray          # bool (z, y, x): inside the dark CSF region


def physical_grid(img):
    """Physical x, y, z (mm) of every voxel center, each as a (z, y, x) array."""
    nx, ny, nz = img.GetSize()
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    idx = np.stack([i, j, k], axis=-1).reshape(-1, 3).T.astype(float)
    d = np.array(img.GetDirection()).reshape(3, 3)
    pts = np.array(img.GetOrigin())[:, None] + d @ (np.array(img.GetSpacing())[:, None] * idx)
    return [p.reshape(nz, ny, nx) for p in pts]


def ellipsoid(x, y, z, center, radii):
    return (((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2
            + ((z - center[2]) / radii[2]) ** 2) <= 1.0


def add_noise(img, sd, seed):
    """Rician-style noise (magnitude of signal + complex Gaussian noise), so values stay >= 0."""
    rng = np.random.default_rng(seed)
    a = sitk.GetArrayFromImage(img)
    noisy = np.sqrt((a + rng.normal(0, sd, a.shape)) ** 2 + rng.normal(0, sd, a.shape) ** 2)
    out = sitk.GetImageFromArray(noisy.astype(np.float32))
    out.CopyInformation(img)
    return out


def make_phantom(seed=0):
    img = sitk.Image([SIZE] * 3, sitk.sitkFloat32)
    img.SetSpacing([SPACING_MM] * 3)
    img.SetOrigin(ORIGIN_MM)
    x, y, z = physical_grid(img)

    head = ellipsoid(x, y, z, HEAD_CENTER_MM, HEAD_SEMI_AXES_MM)
    a = np.where(head, TISSUE, 0.0)
    for offset, r, value in BLOBS:
        c = np.add(HEAD_CENTER_MM, offset)
        a[ellipsoid(x, y, z, c, (r, r, r))] = value
    csf = ellipsoid(x, y, z, np.add(HEAD_CENTER_MM, CSF_CENTER_MM), CSF_RADII_MM)
    a[csf] = CSF

    clean = sitk.GetImageFromArray(a.astype(np.float32))
    clean.CopyInformation(img)
    # Slightly soften the edges, like a real scanner's point-spread, so the optimizer sees
    # smooth intensity changes rather than single-voxel steps.
    clean = sitk.SmoothingRecursiveGaussian(clean, 1.5)
    return Phantom(add_noise(clean, NOISE_SD, seed), head, csf)


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()


@pytest.fixture
def oblique_image():
    """A small random image with a tilted orientation, non-cubic voxels and an offset origin,
    for checking that geometry (origin, spacing, direction) is handled correctly."""
    rng = np.random.default_rng(1)
    img = sitk.GetImageFromArray(rng.uniform(0, 100, (20, 30, 25)).astype(np.float32))
    img.SetSpacing((1.1, 0.9, 2.5))
    img.SetOrigin((12.0, -40.0, 7.5))
    rot = sitk.Euler3DTransform()
    rot.SetRotation(0.3, -0.2, 0.5)
    img.SetDirection(rot.GetMatrix())
    return img
