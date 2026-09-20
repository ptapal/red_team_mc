"""Filesystem roots for the project.

All dataset locations derive from a single root so the code runs on any
machine. By default the root is the ``data/`` directory next to this file,
which is where the datasets live in a fresh clone. Point the ``RTMC_DATA``
environment variable somewhere else to override it, for example::

export RTMC_DATA=/scratch/eeg/red_team_mc_data
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("RTMC_DATA", REPO_ROOT / "data"))

RAW_ROOT = DATA_ROOT / "raw"
PREP_ROOT = DATA_ROOT / "preprocessed"
