"""Repeat 64 mT sessions: how stable are bedside scans of the same person on different days?

    python repeat_sessions.py         # about 5 minutes (5 subjects, 10 repeat sessions)

Five of the paired volunteers had the 64 mT scan repeated in later sessions. For each repeat
session N (64 mT T2 every time):

    S:   64 mT session 1 --> 64 mT session N        same scanner, same contrast
         -> how far the head sat from its session-1 position, in scanner coordinates

and a second consistency loop, like consistency_loop.py but through a different day:

    path 1:  64 mT session 1 --A1--> 3T T1
    path 2:  64 mT session 1 --S--> 64 mT session N --AN--> 3T T1

A1 and AN are separate 64 mT -> 3T registrations of scans made on different days. If both are
right, the two paths agree. Unlike the 3T T2 loop, the two 64 mT scans have independent noise
and the head sits differently in the scanner each time, so this loop is not fooled by
something that is fixed in one particular 64 mT scan.

Writes results/repeat_sessions.json.
"""
import json

import numpy as np
import SimpleITK as sitk

from consistency_loop import rotation_deg
from register import (OUT, load_3t, load_lowfield, lowfield_sessions, paired_subjects,
                      register_images, registration_error_mm)


def check_subject(subject, sessions):
    t1 = load_3t(subject, "T1w")
    low1 = load_lowfield(subject, sessions[0], "T2w")
    print(f"{subject}: A1  64 mT {sessions[0]} -> 3T T1")
    fixed1, _, A1, _, conv_a1 = register_images(low1, t1)
    rows = []
    for ses in sessions[1:]:
        lowN = load_lowfield(subject, ses, "T2w")
        print(f"{subject}: S   64 mT {sessions[0]} -> {ses}")
        _, _, S, mi_s, conv_s = register_images(low1, lowN)
        print(f"{subject}: AN  64 mT {ses} -> 3T T1")
        _, _, AN, _, conv_an = register_images(lowN, t1)
        loop_mean, loop_max = registration_error_mm(fixed1, A1, sitk.CompositeTransform([AN, S]))
        move_mean, move_max = registration_error_mm(fixed1, S, sitk.Euler3DTransform())
        rows.append({
            "subject": subject, "session": ses,
            "head_position_change": {"mean_mm": round(move_mean, 1), "max_mm": round(move_max, 1),
                                     "rotation_deg": round(rotation_deg(S), 1)},
            "loop_error_mm": {"mean": round(loop_mean, 2), "max": round(loop_max, 2)},
            "mi_session1_to_session": round(mi_s, 4),
            "converged": bool(conv_a1 and conv_s and conv_an),
        })
        print(f"== {subject} {ses}: head moved {move_mean:.1f} mm, loop error {loop_mean:.2f} mm\n")
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for subject in paired_subjects():
        sessions = lowfield_sessions(subject)
        if len(sessions) > 1:
            rows += check_subject(subject, sessions)
    (OUT / "repeat_sessions.json").write_text(json.dumps(rows, indent=2))
    loop = np.array([r["loop_error_mm"]["mean"] for r in rows])
    print(f"Repeat-session loop error over {len(rows)} sessions: median {np.median(loop):.2f} mm, "
          f"max {loop.max():.2f} mm")


if __name__ == "__main__":
    main()
