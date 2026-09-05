"""
Shared fixtures.

Two things happen here and nothing else. src/ goes on sys.path, because the
scripts import each other by bare module name (`import dataprep as dp`) rather
than as a package, and the cohort is loaded once per session, because reading two
XPT files takes a couple of seconds and every test that touches real data wants
the same 12,156 rows.

What these tests are for: the invariants that no single run can check. A pipeline
run proves the code executed; these prove the identities hold, the split is
disjoint, the WHO bands are the WHO bands, a truncated download is refused, and
the result JSONs still carry the keys the figures and the README read out of them.
The last of those is the one that catches the state this project was actually
found in - source newer than results.

Run:  python -m pytest tests -q
      python -m pytest tests -q -m "not slow"     skip the ones that read data
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

ROOT = SRC.parent
RESULTS = ROOT / "results"


@pytest.fixture(scope="session")
def dp():
    import dataprep
    return dataprep


@pytest.fixture(scope="session")
def cohort(dp):
    """The one cohort every script shares: (df, attrition)."""
    return dp.load_cohort()


@pytest.fixture(scope="session")
def splits(dp, cohort):
    return dp.split3(cohort[0])


def _load(name):
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not built yet; run python src/run_all.py")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def ablation_json():
    return _load("ablation.json")


@pytest.fixture(scope="session")
def benchmark_json():
    return _load("benchmark.json")


@pytest.fixture(scope="session")
def weighted_json():
    return _load("weighted.json")


@pytest.fixture(scope="session")
def identity_json():
    return _load("identity_audit.json")


@pytest.fixture(scope="session")
def manifest_json():
    return _load("run_manifest.json")
