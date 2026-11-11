"""Independent accuracy check: does the registration agree with itself through a second 3T scan?

    python consistency_loop.py        # about 5 minutes for all 11 subjects

The self-consistency test in register.py re-registers a moved copy of the *same* 3T image, so
it is optimistic. Here the second path uses a *different* 3T image, the T2 from the same 3T
session:

    path 1:  64 mT T2 --A--> 3T T1                     (the main result)
    path 2:  64 mT T2 --B--> 3T T2 --C--> 3T T1         (via the 3T T2)

A and B are separate 64 mT -> 3T registrations. C is a 3T T2 -> 3T T1 registration within
one session on one scanner: both scans are sharp, so C is the easy step, and it also measures
how much the head moved between those two 3T sequences. If everything were exact, both paths
would send every point of the head to the same place. The loop error is how far apart they put
it (mean and worst, over the head in the 64 mT scan).

The loop error still isn't ground truth: both paths start from the same 64 mT scan, so a
distortion in that scan would bend both the same way and stay hidden. It does catch errors
that depend on which 3T image the scan is registered to.

Writes results/consistency_loop.json.
"""
import json
import math

import numpy as np
import SimpleITK as sitk

from register import (OUT, load_3t, load_lowfield, paired_subjects, register_images,
                      registration_error_mm)


def rotation_deg(tf):
    """Total rotation angle of a rigid transform (degrees)."""
    if tf.GetName() == "CompositeTransform":
        tf = sitk.CompositeTransform(tf).GetNthTransform(0)
    R = np.array(sitk.Euler3DTransform(tf).GetMatrix()).reshape(3, 3)
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def check_subject(subject):
    low = load_lowfield(subject, contrast="T2w")
    t1 = load_3t(subject, "T1w")
    t2 = load_3t(subject, "T2w")

    print(f"{subject}: A  64 mT T2 -> 3T T1")
    fixed, _, A, mi_a = register_images(low, t1)
    print(f"{subject}: B  64 mT T2 -> 3T T2")
    _, _, B, mi_b = register_images(low, t2)
    print(f"{subject}: C  3T T2 -> 3T T1 (same session)")
    fixed_t2, _, C, mi_c = register_images(t2, t1)

    # CompositeTransform([C, B]) applies B first, then C: 64 mT point -> 3T T2 -> 3T T1.
    via_t2 = sitk.CompositeTransform([C, B])
    loop_mean, loop_max = registration_error_mm(fixed, A, via_t2)
    move_mean, move_max = registration_error_mm(fixed_t2, C, sitk.Euler3DTransform())
    return {
        "subject": subject,
        "loop_error_mm": {"mean": round(loop_mean, 2), "max": round(loop_max, 2)},
        "within_3t_session_motion": {
            "mean_mm": round(move_mean, 2), "max_mm": round(move_max, 2),
            "rotation_deg": round(rotation_deg(C), 2),
        },
        "mutual_information": {"A_lowfield_to_3t_t1": round(mi_a, 4),
                               "B_lowfield_to_3t_t2": round(mi_b, 4),
                               "C_3t_t2_to_3t_t1": round(mi_c, 4)},
    }


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for subject in paired_subjects():
        r = check_subject(subject)
        rows.append(r)
        print(f"== {subject}: loop error {r['loop_error_mm']['mean']} mm "
              f"(worst {r['loop_error_mm']['max']} mm); 3T T1/T2 head motion "
              f"{r['within_3t_session_motion']['mean_mm']} mm\n")
    (OUT / "consistency_loop.json").write_text(json.dumps(rows, indent=2))
    loop = np.array([r["loop_error_mm"]["mean"] for r in rows])
    print(f"Loop error over {len(rows)} subjects: median {np.median(loop):.2f} mm, "
          f"max {loop.max():.2f} mm")


if __name__ == "__main__":
    main()
