"""Loader/roundtrip tests for analysis/recalibrate_mcd.py.

The recalibration math lives in src/uncertainty.py (tested in test_uncertainty.py); this
file covers reading the folded-in MC-dropout from the validation npz archive.
"""

import io
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.recalibrate_mcd import load_run_long, recalibrate_run


def _write_validation_zip(tmp_path, experiment, n_anchors=120, horizon=8,
                          true_sigma=10.0, reported_sigma=2.0, seed=0):
    """Write a validation/best/npz.zip with folded-in MCD (mcd_mean/mcd_std/targets)."""
    rng = np.random.default_rng(seed)
    run_dir = tmp_path / experiment / "validation" / "best"
    run_dir.mkdir(parents=True)
    base = np.datetime64("2022-01-01T00:00:00")
    with zipfile.ZipFile(run_dir / "npz.zip", "w") as z:
        for a in range(n_anchors):
            ts = base + np.timedelta64(30 * a, "m")
            stem = np.datetime_as_string(ts, unit="s").replace("-", "").replace(":", "").replace("T", "")
            mu = rng.normal(50.0, 20.0, size=horizon)
            obs = mu + rng.normal(0.0, true_sigma, size=horizon)
            buf = io.BytesIO()
            np.savez(
                buf,
                anchor=stem,
                targets=obs.reshape(horizon, 1),          # (target_len, n_vars)
                mcd_mean=mu.reshape(horizon, 1),
                mcd_std=np.full((horizon, 1), reported_sigma),
            )
            z.writestr(f"{stem}.npz", buf.getvalue())
    return run_dir


class TestLoadRun:
    def test_roundtrip_and_recalibrate(self, tmp_path):
        exp = "in12h_out12h_gnn_patchtst"
        run_dir = _write_validation_zip(tmp_path, exp, n_anchors=120, horizon=8)

        long = load_run_long(str(tmp_path), exp)
        assert long["true"].size == 120 * 8
        assert np.all(long["anchor"][:-1] <= long["anchor"][1:])  # sorted in time

        result = recalibrate_run(str(tmp_path), exp)
        assert result["picp_2sigma_raw"] < 0.6
        assert result["picp_2sigma_recal"] == pytest.approx(0.95, abs=0.05)
        assert (run_dir / "calibration.json").exists()

    def test_missing_archive_returns_none(self, tmp_path):
        assert recalibrate_run(str(tmp_path), "nonexistent_run") is None

    def test_validation_npz_without_mcd_is_skipped(self, tmp_path):
        """A plain validation npz (no folded MCD) yields no recalibration."""
        exp = "in6h_out6h_linear"
        run_dir = tmp_path / exp / "validation" / "best"
        run_dir.mkdir(parents=True)
        with zipfile.ZipFile(run_dir / "npz.zip", "w") as z:
            buf = io.BytesIO()
            np.savez(buf, targets=np.zeros((8, 1)), predictions=np.zeros((8, 1)))
            z.writestr("20220101000000.npz", buf.getvalue())
        assert recalibrate_run(str(tmp_path), exp) is None
