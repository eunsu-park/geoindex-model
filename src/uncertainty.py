"""Monte Carlo Dropout uncertainty: per-event sampling + calibration + recalibration.

Canonical uncertainty library, shared by the validation pass (``src/validators.py``,
which folds MC-dropout into validation) and the post-hoc recalibration script
(``analysis/recalibrate_mcd.py``). MC-dropout keeps the model in eval but re-enables every
``nn.Dropout`` so each forward samples; ``num_samples`` stochastic passes give a predictive
distribution per event. Summary stats are computed on the **denormalized** samples, with the
non-negative clip gated by ``Normalizer.is_nonnegative`` so signed targets (Dst/SYM-H) keep
their sign.

Calibration note: raw MC-dropout intervals are under-dispersed (they capture only epistemic
uncertainty), so ``recalibrate_cv`` fits one positive ``sigma_scale`` per run to restore ~95%
coverage at 2 sigma. Point-forecast metrics are never touched.

Metric/recalibration math ported from the MAGIA project (``magia/eval/{evaluate,recalibrate}.py``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf

# Empirical quantiles stored per event, in ascending order.
QUANTILE_LEVELS = (0.025, 0.05, 0.5, 0.95, 0.975)
QUANTILE_KEYS = ("mcd_q025", "mcd_q05", "mcd_median", "mcd_q95", "mcd_q975")


def enable_dropout(model: nn.Module) -> int:
    """Re-enable every ``nn.Dropout`` (keep the rest in eval) for MC sampling.

    Args:
        model: Model already switched to eval mode.

    Returns:
        Number of dropout layers switched to train mode.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            count += 1
    return count


def mcd_sample_stats(model, inputs, sdo, normalizer, target_variables,
                     num_samples: int = 100) -> dict:
    """Run ``num_samples`` MC-dropout passes and summarize the denormalized predictive.

    Every returned stat array has shape ``(batch, target_len, n_target_vars)`` in the
    original index scale. Per-sample denormalization matches the deterministic path, and
    non-negative clipping is applied only for non-negative targets (``is_nonnegative``).

    Args:
        model: The trained model (will be left in eval mode on return).
        inputs: Input tensor already on the model device.
        sdo: Optional image tensor (None in the CSV timeseries path).
        normalizer: Normalizer with ``denormalize_omni`` and ``is_nonnegative``.
        target_variables: Ordered target variable names (one per output channel).
        num_samples: Number of stochastic forward passes.

    Returns:
        Dict with mcd_mean, mcd_std, mcd_min, mcd_max, mcd_median, mcd_q025, mcd_q05,
        mcd_q95, mcd_q975 (each an array), and n_samples (int).
    """
    model.eval()
    enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model(inputs, sdo, return_features=False).detach().cpu().numpy()
            denorm = np.empty(out.shape, dtype=np.float64)
            for v, var in enumerate(target_variables):
                d = normalizer.denormalize_omni(out[..., v], var)
                if normalizer.is_nonnegative(var):
                    d = np.clip(d, 0.0, None)
                denorm[..., v] = d
            samples.append(denorm)
    model.eval()  # restore: dropout off so later deterministic use is unaffected

    arr = np.stack(samples, axis=0)  # (S, batch, target_len, n_vars)
    quantiles = np.quantile(arr, QUANTILE_LEVELS, axis=0)  # (5, batch, target_len, n_vars)
    stats = {
        "mcd_mean": arr.mean(axis=0),
        "mcd_std": arr.std(axis=0, ddof=1) if num_samples > 1 else np.zeros(arr.shape[1:]),
        "mcd_min": arr.min(axis=0),
        "mcd_max": arr.max(axis=0),
        "n_samples": int(num_samples),
    }
    for key, q in zip(QUANTILE_KEYS, quantiles):
        stats[key] = q
    return stats


def uncertainty_metrics(true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """Calibration/sharpness metrics for a Gaussian predictive (mean, std).

    PICP = prediction-interval coverage probability (fraction of ``true`` within
    ``mean +/- n*sigma``; well-calibrated ~ 0.68 / 0.95). Sharpness = mean 2 sigma
    interval width. NLL/CRPS are proper scores under the Gaussian assumption (lower better).

    Args:
        true: Ground-truth values (any shape; flattened).
        mean: Predictive mean (same shape as ``true``).
        std: Predictive standard deviation (same shape as ``true``).

    Returns:
        Dict of picp_1sigma, picp_2sigma, sharpness_2sigma, nll_gaussian,
        crps_gaussian, mae_mcd_mean, mcd_std_mean.
    """
    true = np.asarray(true, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.clip(np.asarray(std, dtype=np.float64).ravel(), 1e-6, None)
    z = (true - mean) / std
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    nll = 0.5 * np.log(2.0 * np.pi * std ** 2) + 0.5 * z ** 2
    return {
        "picp_1sigma": float(np.mean(np.abs(z) <= 1.0)),
        "picp_2sigma": float(np.mean(np.abs(z) <= 2.0)),
        "sharpness_2sigma": float(np.mean(4.0 * std)),
        "nll_gaussian": float(np.mean(nll)),
        "crps_gaussian": float(np.mean(crps)),
        "mae_mcd_mean": float(np.mean(np.abs(mean - true))),
        "mcd_std_mean": float(np.mean(std)),
    }


def fit_sigma_scale(true, mean, std, coverage: float = 0.95, k: float = 2.0) -> float:
    """Coverage-matching scale: ``s`` so ``mean +/- k*(s*std)`` covers ``coverage`` of truths.

    With ``z = (true - mean) / std``, the band ``|true - mean| <= k*s*std`` covers
    ``coverage`` when ``k*s = quantile(|z|, coverage)``. Returns
    ``s = quantile(|z|, coverage) / k``.

    Args:
        true: Ground-truth values.
        mean: Predictive mean.
        std: Predictive standard deviation.
        coverage: Target coverage fraction (default 0.95).
        k: Sigma multiple the coverage is matched at (default 2.0).

    Returns:
        Positive scalar sigma scale.
    """
    std = np.clip(np.asarray(std, dtype=np.float64), 1e-6, None)
    z = np.abs((np.asarray(true, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / std)
    return float(np.quantile(z, coverage) / k)


def recalibrate_cv(anchor, horizon, true, mean, std,
                   n_folds: int = 5, gap=None,
                   coverage: float = 0.95, k: float = 2.0) -> dict:
    """Temporal K-fold cross-fit of the per-run sigma scale; pooled out-of-fold metrics.

    Anchors are split into ``n_folds`` contiguous time blocks (random folds would leak:
    adjacent anchors share overlapping target windows). For each fold the scale is fit on
    the other folds -- excluding a ``gap``-anchor band on each side of the held-out block
    to drop boundary target-window overlap -- then applied to the held-out fold. The
    pooled out-of-fold recalibrated std feeds ``uncertainty_metrics``. ``sigma_scale`` is
    the deploy value (fit on all anchors); ``sigma_scale_folds`` exposes drift across the
    solar cycle.

    Args:
        anchor: Per-row anchor timestamp (datetime64 or sortable), shape (N,).
        horizon: Per-row forecast horizon index, shape (N,).
        true: Per-row ground truth, shape (N,).
        mean: Per-row MC-dropout mean, shape (N,).
        std: Per-row MC-dropout std, shape (N,).
        n_folds: Number of contiguous temporal folds (default 5).
        gap: Anchor-band to exclude on each side of a held-out block. Defaults to the
            target-window length (max horizon + 1) as a conservative proxy; correct when
            the anchor cadence equals the target cadence.
        coverage: Target coverage fraction (default 0.95).
        k: Sigma multiple the coverage is matched at (default 2.0).

    Returns:
        Dict with deploy sigma_scale, per-fold scales, and raw vs. recalibrated
        PICP/sharpness/NLL/CRPS.
    """
    true = np.asarray(true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    horizon = np.asarray(horizon)
    max_h = int(horizon.max())
    gap = (max_h + 1) if gap is None else int(gap)

    anchor_vals = np.asarray(anchor)
    anchors = np.unique(anchor_vals)  # sorted unique anchors (time order)
    pos = np.searchsorted(anchors, anchor_vals)
    bounds = np.linspace(0, len(anchors), n_folds + 1).astype(int)

    s_folds, recal_std = [], np.empty(len(true), dtype=np.float64)
    for f in range(n_folds):
        lo, hi = bounds[f], bounds[f + 1]  # held-out fold = anchor positions [lo, hi)
        test = (pos >= lo) & (pos < hi)
        train = (pos < lo - gap) | (pos >= hi + gap)
        if not train.any():  # degenerate (too few anchors for the gap): fall back to all
            train = ~test
        s = fit_sigma_scale(true[train], mean[train], std[train], coverage, k)
        s_folds.append(s)
        recal_std[test] = s * std[test]

    recal = uncertainty_metrics(true, mean, recal_std)
    raw = uncertainty_metrics(true, mean, std)
    return {
        "method": f"coverage_match_{k:g}sigma",
        "n_folds": n_folds,
        "gap": gap,
        "coverage": coverage,
        "n_anchors": int(len(anchors)),
        "n_points": int(len(true)),
        "sigma_scale": fit_sigma_scale(true, mean, std, coverage, k),
        "sigma_scale_folds": [float(x) for x in s_folds],
        "sigma_scale_std": float(np.std(s_folds)),
        "picp_2sigma_raw": raw["picp_2sigma"],
        "picp_1sigma_raw": raw["picp_1sigma"],
        "crps_gaussian_raw": raw["crps_gaussian"],
        "picp_1sigma_recal": recal["picp_1sigma"],
        "picp_2sigma_recal": recal["picp_2sigma"],
        "sharpness_2sigma_recal": recal["sharpness_2sigma"],
        "nll_gaussian_recal": recal["nll_gaussian"],
        "crps_gaussian_recal": recal["crps_gaussian"],
    }
