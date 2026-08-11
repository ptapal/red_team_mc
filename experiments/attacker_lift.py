import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import csv

RESULTS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'results')

STREAMS = [
    ("Zhang",      "RSVP",  "zhang/threshold_sweep_std.csv",       "zhang/adv_far_std.csv"),
    ("Won",        "RSVP",  "won/threshold_sweep_std.csv",         "won/adv_far_std.csv"),
    ("Wang-EO",    "Rest",  "wang/threshold_sweep_std_eo.csv",     "wang/adv_far_std_eo.csv"),
    ("Wang-EC",    "Rest",  "wang/threshold_sweep_std_ec.csv",     "wang/adv_far_std_ec.csv"),
    ("COG-BCI-EO", "Rest",  "COG-BCI/threshold_sweep_std_eo.csv",  "COG-BCI/adv_far_std_eo.csv"),
    ("COG-BCI-EC", "Rest",  "COG-BCI/threshold_sweep_std_ec.csv",  "COG-BCI/adv_far_std_ec.csv"),
    ("Cho",        "MI",    "cho/threshold_sweep_std.csv",         "cho/adv_far_std.csv"),
    ("Lee",        "MI",    "lee/threshold_sweep_std.csv",         "lee/adv_far_std.csv"),
]


def read(path):
    with open(_os.path.join(RESULTS, path)) as fh:
        return list(csv.DictReader(fh))


def interp(rows, p, col):
    xs = [float(r["p"]) for r in rows]
    ys = [float(r[col]) for r in rows]
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= p <= xs[i + 1]:
            t = (p - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])


rows_out = []
print(f"{'stream':<12}{'para':<6}{'p':>4}{'base':>8}{'adv':>8}{'lift':>9}")
for name, para, base_f, adv_f in STREAMS:
    base_rows, adv_rows = read(base_f), read(adv_f)
    for r in adv_rows:
        p    = float(r["p"])
        base = interp(base_rows, p, "FAR_mean")
        adv  = float(r["FAR_mean"])
        exact = any(float(b["p"]) == p for b in base_rows)
        rows_out.append(dict(stream=name, paradigm=para, p=int(p),
                             base_far=round(base, 4), adv_far=round(adv, 4),
                             lift=round(adv - base, 4),
                             base_interpolated=(not exact)))
        print(f"{name:<12}{para:<6}{int(p):>4}{base:>8.3f}{adv:>8.3f}{adv - base:>+9.3f}")
    print()

out = _os.path.join(RESULTS, "attacker_lift.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

rest = [r for r in rows_out if r["paradigm"] == "Rest"]
neg  = [r for r in rest if r["lift"] < 0]
print(f"Resting-state: lift < 0 in {len(neg)}/{len(rest)} cells, "
      f"range [{min(r['lift'] for r in rest):+.3f}, {max(r['lift'] for r in rest):+.3f}]")
print(f"Wrote {out}")
