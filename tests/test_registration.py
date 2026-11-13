"""Registration on the synthetic phantom: recovering a known movement, picking the better of
the two starting guesses, and the full run_pair pipeline."""
import numpy as np
import pytest
import SimpleITK as sitk

import register
from conftest import add_noise
from register import (euler, image_center, mask_background, mutual_information, normalize,
                      register_best, registration_error_mm, resample_spacing, run_pair)

KNOWN_ROTATION_DEG = (8.0, -5.0, 12.0)
KNOWN_SHIFT_MM = (6.0, -9.0, 4.0)


@pytest.fixture(scope="module")
def head(phantom):
    """The phantom with its background masked, as load_lowfield / load_3t_t1 do for real scans."""
    return mask_background(phantom.image)


@pytest.fixture(scope="module")
def moved_pair(head):
    """(fixed, moving, true transform): fixed is the head moved by a known rotation + shift.

    moved(x) = head(P(x)), so the transform that maps fixed points onto moving points is P."""
    P = euler(KNOWN_ROTATION_DEG, KNOWN_SHIFT_MM, image_center(head))
    moved = sitk.Resample(head, head, P, sitk.sitkLinear, 0.0)
    return normalize(moved), normalize(head), P


def test_register_best_recovers_known_movement(moved_pair):
    fixed, moving, true_tf = moved_pair
    mi, _, est_tf, _, _, _, converged = register_best(fixed, moving)

    mean_mm, max_mm = registration_error_mm(fixed, est_tf, true_tf)
    assert converged
    assert mean_mm < 1.0, f"mean error {mean_mm:.3f} mm"
    assert max_mm < 1.5, f"max error {max_mm:.3f} mm"
    # and it is a real improvement over doing nothing
    assert mi > mutual_information(fixed, moving, sitk.Euler3DTransform())


@pytest.mark.parametrize("correct_start", ["GEOMETRY", "MOMENTS"])
def test_register_best_keeps_start_with_higher_mi(moved_pair, monkeypatch, correct_start):
    """Replace register() so one start lands on the true answer and the other on a wrong one;
    register_best must keep the true answer, whichever start produced it."""
    fixed, moving, true_tf = moved_pair
    wrong_tf = euler((-15.0, 10.0, 20.0), (20.0, 15.0, -10.0), image_center(fixed))
    starts = iter(["GEOMETRY", "MOMENTS"])  # register_best tries them in this order

    def fake_register(fixed, moving, init, **kwargs):
        tf = true_tf if next(starts) == correct_start else wrong_tf
        return sitk.CompositeTransform([tf]), [0.0], True

    monkeypatch.setattr(register, "register", fake_register)
    mi, how, tf, _, starts_differ_mm, start_mi, _ = register_best(fixed, moving)

    assert how == correct_start
    assert registration_error_mm(fixed, tf, true_tf)[0] < 1e-6
    assert mi == pytest.approx(max(start_mi.values()), abs=1e-4)
    assert start_mi[correct_start] > max(v for k, v in start_mi.items() if k != correct_start)
    assert starts_differ_mm > 5.0  # the two starts really did end in different places


def test_run_pair_end_to_end(head):
    """Same pass rule as smoke_test.py, on a phantom pair instead of a real subject.

    Fine image ("3T"): the phantom at 2 mm. Coarse image ("64 mT"): the head moved by a known
    amount, resampled to 3 x 3 x 6 mm voxels and given extra noise."""
    t1 = head
    P = euler((-7.0, 4.0, -10.0), (-5.0, 8.0, 3.0), image_center(head))
    moved = sitk.Resample(head, head, P, sitk.sitkLinear, 0.0)
    low = mask_background(add_noise(resample_spacing(moved, (3.0, 3.0, 6.0)), 0.06, seed=2))

    results, r = run_pair(low, t1, "phantom low-field")

    mi = results["mutual_information"]
    err = results["self_consistency"]["mean_error_mm"]["after"]
    assert mi["after"] > mi["before"]
    assert err < 1.5, f"self-consistency error {err} mm"
    assert results["converged"]
    # On the phantom the true answer is known too, so check it directly as well.
    assert registration_error_mm(r["fixed"], r["est_tf"], P)[0] < 1.5
