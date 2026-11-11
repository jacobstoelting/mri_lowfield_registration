"""Additional check: is registration hard because of big voxels, or because of 64 mT physics?

    python field_strength.py          # about 25 minutes

The 3T session also recorded each scan at a low resolution chosen to match the 64 mT scans
(1.38 x 1.38 x 5 mm, vs 1.6 x 1.6 x 5 mm at 64 mT). Registering those 3T low-res scans to
the 3T high-res T1 with exactly the same pipeline as the 64 mT scans gives a control: the
voxels are about the same size, but the field strength is 3T (more signal, less noise,
standard contrast). If the 64 mT errors are similar, the coarse voxels are what limits
registration; if they are clearly larger, 64 mT physics adds to it.

Caveat: the 3T low-res and high-res scans come from the same session on the same scanner,
so they already sit in one coordinate frame. The registration still starts from the same
starting guesses as for 64 mT (image centers / centers of mass), so that doesn't help it.

Writes results/field_strength.json.
"""
import json

import numpy as np

from all_subjects import CONTRASTS
from register import OUT, load_3t, load_3t_t1, paired_subjects, run_pair


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for subject in paired_subjects():
        t1 = load_3t_t1(subject)
        for contrast in CONTRASTS:
            print(f"== {subject} 3T low-res {contrast}")
            results, _ = run_pair(load_3t(subject, contrast, "lowres"), t1,
                                  f"3T low-res {contrast} (Philips Achieva)")
            rows.append({"subject": subject, "contrast": contrast, **results})
            print(f"== {subject} {contrast}: self-consistency "
                  f"{results['self_consistency']['mean_error_mm']['after']} mm, "
                  f"converged {results['converged']}\n")
    (OUT / "field_strength.json").write_text(json.dumps(rows, indent=2))
    for contrast in CONTRASTS:
        sc = [r["self_consistency"]["mean_error_mm"]["after"] for r in rows if r["contrast"] == contrast]
        print(f"3T low-res {contrast}: self-consistency median {np.median(sc):.2f} mm, max {max(sc):.2f} mm")


if __name__ == "__main__":
    main()
