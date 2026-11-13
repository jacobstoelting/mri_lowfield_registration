"""Runs smoke_test.py on the real Leiden dataset. Skipped when the data is not downloaded
(always the case on GitHub Actions; get it locally with `python download_data.py`)."""
from pathlib import Path

import pytest

import register
import smoke_test

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / register.lowfield_path(register.SUBJECT)

pytestmark = pytest.mark.skipif(not SCAN.exists(), reason="real dataset not in data/ "
                                "(run `python download_data.py`)")


def test_smoke_test_on_real_subject(monkeypatch, capsys):
    monkeypatch.chdir(REPO)  # register.py uses paths relative to the repo root
    smoke_test.main()        # calls sys.exit with a message on failure, which fails the test
    assert "smoke test passed" in capsys.readouterr().out
