"""Quick check that registration still works: one subject, about 30 seconds.

    python smoke_test.py

Exits with an error if the 3T -> 64 mT registration of SUBJECT (register.py) no longer
improves mutual information or its self-consistency error is above LIMIT_MM. Run it after
changing the pipeline, before rerunning all subjects.
"""
import sys

from register import SUBJECT, run_subject

LIMIT_MM = 1.5


def main():
    results, _ = run_subject(SUBJECT)
    mi = results["mutual_information"]
    err = results["self_consistency"]["mean_error_mm"]["after"]
    problems = []
    if mi["after"] <= mi["before"]:
        problems.append(f"mutual information did not improve ({mi['before']} -> {mi['after']})")
    if err > LIMIT_MM:
        problems.append(f"self-consistency error {err} mm is above {LIMIT_MM} mm")
    if problems:
        sys.exit(f"SMOKE TEST FAILED for {SUBJECT}: " + "; ".join(problems))
    print(f"smoke test passed: {SUBJECT}, MI {mi['before']} -> {mi['after']}, "
          f"self-consistency {err} mm (limit {LIMIT_MM} mm)")


if __name__ == "__main__":
    main()
