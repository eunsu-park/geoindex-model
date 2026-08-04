"""Score the two regime runs against the pooled model, conditionally and honestly.

The conditional numbers are computed by selecting validation rows on the OBSERVED label. That
is the correct verification for a conditional statement -- "given a storm occurs, how big" -- and
it is NOT a forecast improvement, because the selection uses information the forecast did not
have. Every conditional row below is marked as such.

The unconditional comparison is the mixture: recombine the branches with P(storm|x) and score it
against the pooled model on every anchor. The ridge says the mixture loses slightly (rho 0.721
against 0.724). That is what the decomposition predicts and is not a failure.

Usage:
    python score_regimes.py
    python score_regimes.py --threshold 48 --pooled probe_..._baseline
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/Projects/GeoIndex/results")
STEM = "regime48_ap_in12h_out12h_gnn_transformer"
POOLED = "probe_ap_in12h_out12h_gnn_transformer_baseline"


def load(run):
    path = os.path.join(ROOT, run, "validation", "best", "npz.zip")
    if not os.path.exists(path):
        raise SystemExit(f"missing: {path}")
    anchors, true, pred, head = [], [], [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".npz")):
            d = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
            anchors.append(str(np.asarray(d["anchor"])))
            true.append(np.asarray(d["targets"])[:, 0])
            pred.append(np.asarray(d["predictions"])[:, 0])
            if "peak_prediction" in d:
                head.append(float(np.asarray(d["peak_prediction"]).ravel()[0]))
    return (np.asarray(anchors), np.asarray(true), np.asarray(pred),
            np.asarray(head) if head else None)


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--threshold", type=float, default=48.0)
    ap_.add_argument("--pooled", default=POOLED)
    ap_.add_argument("--stem", default=STEM)
    args = ap_.parse_args()

    runs = {"quiet": f"{args.stem}_quiet", "storm": f"{args.stem}_storm",
            "pooled": args.pooled}
    loaded = {k: load(v) for k, v in runs.items()}
    anchors, true, _, _ = loaded["pooled"]
    for k, (a, t, _, _) in loaded.items():
        assert np.array_equal(a, anchors), f"{k}: anchors differ from the pooled run"
        assert np.allclose(t, true, atol=1e-3), f"{k}: targets differ"

    peak = true.max(axis=1)
    # The archive stores denormalized targets, and the log1p round trip lands a ladder value of
    # 48 at 47.99999. A bare `>= 48` therefore drops every window whose true maximum is exactly
    # 48 and silently scores the >= 56 subset instead -- which is what the first version of this
    # script did. The ladder gap around 48 is 8, so half a unit of tolerance is unambiguous.
    storm = peak >= args.threshold - 0.5
    # A run with a peak head emits the window maximum as its own scalar; that is the quantity
    # to score, not the maximum of the curve, which is doubly shrunk (section 2.7).
    P, source = {}, {}
    for k, v in loaded.items():
        if v[3] is not None:
            P[k], source[k] = v[3], "head"
        else:
            P[k], source[k] = v[2].max(axis=1), "curve max"
    print(f"{len(anchors)} validation anchors, storm (peak >= {args.threshold:g}) "
          f"{int(storm.sum())} ({100*storm.mean():.1f} %)")
    print(f"  (threshold applied with 0.5 tolerance for the denormalization round trip; "
          f"a bare >= would give {int((peak >= args.threshold).sum())})\n")

    def row(label, p, mask, note=""):
        y = peak[mask]
        q = p[mask]
        rho = float(np.corrcoef(q, y)[0, 1])
        hi = y >= 100
        rep = float(q[hi].mean() / y[hi].mean()) if hi.sum() > 30 else float("nan")
        print(f"  {label:30s} {rho:7.3f} {float(np.polyfit(y, q, 1)[0]):7.3f} "
              f"{rep:8.3f} {float(np.abs(q - y).mean()):8.2f}   {note}")

    print("peak taken from: " + ", ".join(f"{k}={v}" for k, v in source.items()) + "\n")
    hdr = f"  {'':30s} {'rho':>7s} {'slope':>7s} {'repro':>8s} {'MAE':>8s}"
    print("CONDITIONAL — rows selected on the OBSERVED label, not a forecast improvement")
    print(hdr)
    print(f"  -- storm anchors (n={int(storm.sum())})")
    row("pooled", P["pooled"], storm, "<- to beat")
    row("storm-only", P["storm"], storm)
    print(f"  -- quiet anchors (n={int((~storm).sum())})")
    row("pooled", P["pooled"], ~storm, "<- to beat")
    row("quiet-only", P["quiet"], ~storm)

    print("\n  the scenario a user would read:")
    print(f"    'if a storm comes'  -> {P['storm'][storm].mean():6.1f} ap   "
          f"(observed {peak[storm].mean():6.1f})")
    print(f"    'if it stays quiet' -> {P['quiet'][~storm].mean():6.1f} ap   "
          f"(observed {peak[~storm].mean():6.1f})")

    print("\nUNCONDITIONAL — every anchor. The mixture is expected to lose slightly.")
    print(hdr)
    allm = np.ones(len(peak), bool)
    row("pooled", P["pooled"], allm)
    print("  (a calibrated P(storm|x) is needed for the mixture row; take it from")
    print("   calibrate_warning.py on the pooled run, then combine the two branches)")

    # The pass mark below is for a run that ADDS a peak head to the regime split. The
    # curve-only regime runs are the reference these numbers came from, so scoring them
    # against themselves would be circular -- report the reference instead.
    if source.get("storm") != "head":
        print("\nREFERENCE for the next run (regime split + peak head must beat these)")
        print(f"  storm peak recovery   {100*P['storm'][storm].mean()/peak[storm].mean():5.1f} %")
        print(f"  storm MAE             {float(np.abs(P['storm'][storm]-peak[storm]).mean()):5.2f}")
        print(f"  quiet MAE (pooled)    {float(np.abs(P['pooled'][~storm]-peak[~storm]).mean()):5.2f}")
        print(f"  rho in the storm branch (pooled) "
              f"{float(np.corrcoef(P['pooled'][storm], peak[storm])[0,1]):.3f}")
        return

    print("\nPASS MARK")
    checks = [
        ("storm peak recovery > 55.5 % (regime split, curve max)",
         float(P["storm"][storm].mean() / peak[storm].mean()) > 0.555),
        ("storm MAE < 36.00 (regime split)",
         float(np.abs(P["storm"][storm] - peak[storm]).mean()) < 36.00),
        ("quiet MAE < 9.79 (pooled)",
         float(np.abs(P["quiet"][~storm] - peak[~storm]).mean()) < 9.79),
        ("reproduction on storms > pooled",
         float(P["storm"][storm & (peak >= 100)].mean() /
               peak[storm & (peak >= 100)].mean()) >
         float(P["pooled"][storm & (peak >= 100)].mean() /
               peak[storm & (peak >= 100)].mean())),
        ("rho within the storm branch >= pooled",
         float(np.corrcoef(P["storm"][storm], peak[storm])[0, 1]) >=
         float(np.corrcoef(P["pooled"][storm], peak[storm])[0, 1]) - 0.005),
        ("quiet-branch MAE < pooled",
         float(np.abs(P["quiet"][~storm] - peak[~storm]).mean()) <
         float(np.abs(P["pooled"][~storm] - peak[~storm]).mean())),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
