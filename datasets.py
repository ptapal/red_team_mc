import numpy as np
from config import SEED
from paths import PREP_ROOT

_rng_won = np.random.default_rng(SEED)
WON_15 = sorted(
    _rng_won.choice([f"S{i:02d}" for i in range(1, 56)], size=15, replace=False).tolist()
)

# 20 Wang subjects subsampled from 60 to match COG-BCI N (seed 42)
_rng_t20 = np.random.default_rng(42)
TANNER_20 = sorted(
    _rng_t20.choice([f"sub-{i:02d}" for i in range(1, 61)], size=20, replace=False).tolist()
)

_rng_cho = np.random.default_rng(42)
CHO_20 = sorted(
    _rng_cho.choice([f"s{i:02d}" for i in range(1, 53)], size=20, replace=False).tolist()
)

_rng_lee = np.random.default_rng(42)
LEE_20 = sorted(
    _rng_lee.choice([f"sub-{i:02d}" for i in range(1, 55)], size=20, replace=False).tolist()
)

COGBCI_SUBS = [f"sub-{i:02d}" for i in range(10, 30)]
TANNER_DAYS = ["session1", "session2", "session3"]
COGBCI_DAYS = ["S1", "S2", "S3"]
CHO_DAYS    = ["session1"]
LEE_DAYS    = ["session1", "session2"]

_DATA = str(PREP_ROOT)

_BASE = {
    "zhang":     dict(label="Zhang (RSVP)",        erp_root=f"{_DATA}/zhang/won_channels", participants=[f"S{i}" for i in range(1, 16)], days=["Day_1", "Day_7", "Day_80", "Day_200"]),
    "won":       dict(label="Won (RSVP)",           erp_root=f"{_DATA}/won",                participants=WON_15,      days=["Day_1"]),
    "tanner_eo": dict(label="Wang-EO (resting)",    erp_root=f"{_DATA}/wang/eo",            participants=TANNER_20,   days=TANNER_DAYS),
    "tanner_ec": dict(label="Wang-EC (resting)",    erp_root=f"{_DATA}/wang/ec",            participants=TANNER_20,   days=TANNER_DAYS),
    "cogbci_eo": dict(label="COG-BCI-EO (resting)", erp_root=f"{_DATA}/cogbci/eo",          participants=COGBCI_SUBS, days=COGBCI_DAYS),
    "cogbci_ec": dict(label="COG-BCI-EC (resting)", erp_root=f"{_DATA}/cogbci/ec",          participants=COGBCI_SUBS, days=COGBCI_DAYS),
    "cho":       dict(label="Cho (motor imagery)",  erp_root=f"{_DATA}/cho",                participants=CHO_20,      days=CHO_DAYS),
    "lee":       dict(label="Lee (motor imagery)",  erp_root=f"{_DATA}/lee",                participants=LEE_20,      days=LEE_DAYS),
}


def get_datasets(prefix="results"):
    return {k: {**v, "out_dir": f"{prefix}_{k}"} for k, v in _BASE.items()}
