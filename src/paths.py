"""
Project paths, resolved from this file's location.

Every script imports from here so that no script depends on the directory it
was launched from. Run `python src/dataprep.py` or `cd src && python dataprep.py`
and both work.
"""

from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# The individual XPT filenames are NOT defined here. download_data.CYCLES owns
# them, for all four cycles at once, so there is one place that knows which
# files the study needs and which weight column each one carries.

# The output directories are created on import, so every script can write
# without each one checking first. The cost is that no caller can ever find one
# missing, which makes a misplaced copy of this file dangerous: dropped into
# another tree it would silently build a second, empty data/models/results/
# figures skeleton there and every script would then look for NHANES files in a
# directory that will never have any. The marker check below makes that fail at
# import with the wrong root named, instead of one stage at a time much later.
#
# Two markers, and the second is the one that matters. dataprep.py sits BESIDE
# this file, so it only proves src/ arrived intact - copy the whole src/ tree
# somewhere else and that check still passes while ROOT is wrong, which is
# exactly the failure it was supposed to catch. requirements.txt sits in ROOT, so
# it is what actually distinguishes "the project root" from "the parent of some
# directory that happens to contain a copy of src/".
_MARKERS = (SRC / "dataprep.py", ROOT / "requirements.txt")
_missing = [m for m in _MARKERS if not m.exists()]
if _missing:
    raise SystemExit(
        f"paths.py: {ROOT} does not look like the project root - expected "
        + " and ".join(str(m.relative_to(ROOT)) for m in _MARKERS)
        + f", and {', '.join(m.name for m in _missing)} is missing. Nothing was "
        f"created. Run the scripts from the checkout that contains src/, or fix "
        f"the copy of paths.py at {__file__}.")

for _d in (DATA, MODELS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
