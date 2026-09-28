"""Conformal prediction intervals.

A Gaussian process reports a standard deviation, but that number is only as
good as the kernel and the noise model. Conformal prediction takes the model's
standard deviation and rescales it using held-out errors, so that a stated 90%
interval really does contain about 90% of new values.

The version here is split conformal with normalised scores: the score is the
absolute error divided by the model's own standard deviation, so the interval
stays wide where the model is unsure and narrow where it is confident.
"""
from __future__ import annotations

import numpy as np


class SplitConformal:
    """Fit on a calibration set the model never trained on."""

    def __init__(self, alpha: float = 0.1, normalised: bool = True):
        self.alpha = alpha
        self.normalised = normalised
        self.q_: float | None = None
        self.n_cal_: int = 0

    def fit(self, y_cal: np.ndarray, mu_cal: np.ndarray, sd_cal: np.ndarray
            ) -> "SplitConformal":
        y_cal = np.asarray(y_cal, dtype=float)
        mu_cal = np.asarray(mu_cal, dtype=float)
        sd_cal = np.clip(np.asarray(sd_cal, dtype=float), 1e-6, None)
        resid = np.abs(y_cal - mu_cal)
        scores = resid / sd_cal if self.normalised else resid
        n = len(scores)
        # the finite-sample correction: without it the interval is a little
        # too narrow on small calibration sets
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level))
        self.n_cal_ = n
        return self

    def interval(self, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("fit the calibrator first")
        mu = np.asarray(mu, dtype=float)
        sd = np.clip(np.asarray(sd, dtype=float), 1e-6, None)
        half = self.q_ * sd if self.normalised else self.q_ * np.ones_like(mu)
        return mu - half, mu + half

    def evaluate(self, y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
        lo, hi = self.interval(mu, sd)
        y = np.asarray(y, dtype=float)
        covered = (y >= lo) & (y <= hi)
        return {
            "alpha": self.alpha,
            "nominal_coverage": 1 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "mean_width": float(np.mean(hi - lo)),
            "median_width": float(np.median(hi - lo)),
            "n": int(len(y)),
            "q": self.q_,
        }


def coverage_by_bin(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    by: np.ndarray, n_bins: int = 5) -> "np.ndarray":
    """Coverage inside bins of some other quantity, for example predicted value.

    Overall coverage can look correct while one part of the range is badly
    served. This splits the check up.
    """
    import pandas as pd
    y, lo, hi, by = map(lambda a: np.asarray(a, dtype=float), (y, lo, hi, by))
    covered = (y >= lo) & (y <= hi)
    bins = pd.qcut(by, n_bins, duplicates="drop")
    df = pd.DataFrame({"covered": covered, "width": hi - lo, "bin": bins})
    return df.groupby("bin", observed=True).agg(
        n=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
    ).reset_index()
