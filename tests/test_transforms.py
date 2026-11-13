"""Transforms: euler() and its inverse undo each other, and a transform survives a file round-trip."""
import numpy as np
import pytest
import SimpleITK as sitk

from register import euler

CENTER = (10.0, -25.0, 40.0)
POINTS = [(0.0, 0.0, 0.0), (55.5, -12.25, 3.0), (-80.0, 64.0, -120.0), CENTER]


@pytest.mark.parametrize("rot_deg, shift_mm", [
    ((6.0, -4.0, 9.0), (7.0, -12.0, 5.0)),   # the self-consistency movement in register.py
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),      # identity
    ((-30.0, 45.0, 170.0), (100.0, 0.0, -50.0)),
])
def test_euler_then_inverse_returns_point(rot_deg, shift_mm):
    t = euler(rot_deg, shift_mm, CENTER)
    inv = t.GetInverse()
    for p in POINTS:
        back = inv.TransformPoint(t.TransformPoint(p))
        assert np.linalg.norm(np.subtract(back, p)) < 1e-6


def test_euler_rotates_about_center():
    """With no shift, the center is a fixed point; with a shift, it moves by exactly that shift."""
    assert np.allclose(euler((10, 20, 30), (0, 0, 0), CENTER).TransformPoint(CENTER), CENTER)
    moved = euler((10, 20, 30), (1.0, 2.0, 3.0), CENTER).TransformPoint(CENTER)
    assert np.allclose(np.subtract(moved, CENTER), (1.0, 2.0, 3.0))


def test_transform_file_round_trip(tmp_path):
    t = euler((6.0, -4.0, 9.0), (7.0, -12.0, 5.0), CENTER)
    path = tmp_path / "t1_to_lowfield.tfm"
    sitk.WriteTransform(t, str(path))
    r = sitk.ReadTransform(str(path))

    assert np.allclose(r.GetParameters(), t.GetParameters(), atol=1e-12)
    assert np.allclose(r.GetFixedParameters(), t.GetFixedParameters(), atol=1e-12)
    for p in POINTS:
        assert np.linalg.norm(np.subtract(r.TransformPoint(p), t.TransformPoint(p))) < 1e-6
