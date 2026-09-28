"""Turning descriptor tables into a matrix a Gaussian process can handle.

A GP with an RBF or Matern kernel wants a modest number of columns on a common
scale. Raw RDKit gives about 200 descriptors, many of them near copies of each
other. This module prunes and standardises, and it learns those choices on the
training set only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """Fit on training molecules, apply to everything else.

    Steps, in order:
      1. drop columns that are NaN anywhere in the training set
      2. drop columns with almost no variance
      3. drop one of every pair of columns correlated above `corr_max`
      4. clip extreme values to training quantiles, so one odd molecule cannot
         stretch the scale
      5. standardise to zero mean and unit variance

    Nothing is imputed. A molecule whose descriptors failed stays NaN and gets
    dropped by the caller.
    """

    def __init__(self, corr_max: float = 0.95, var_min: float = 1e-8,
                 clip_quantile: float = 0.001, max_features: int | None = None):
        self.corr_max = corr_max
        self.var_min = var_min
        self.clip_quantile = clip_quantile
        self.max_features = max_features
        self.columns_: list[str] = []
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        d = df.replace([np.inf, -np.inf], np.nan)
        d = d.loc[:, d.notna().all(axis=0)]
        d = d.loc[:, d.var(axis=0) > self.var_min]

        # correlation pruning, keeping the first column of each correlated pair
        corr = d.corr().abs().to_numpy()
        keep: list[int] = []
        for j in range(corr.shape[1]):
            if all(corr[j, k] <= self.corr_max for k in keep):
                keep.append(j)
        cols = [d.columns[j] for j in keep]

        if self.max_features is not None and len(cols) > self.max_features:
            # keep the highest variance survivors
            v = d[cols].var().sort_values(ascending=False)
            cols = list(v.index[: self.max_features])

        self.columns_ = cols
        sub = d[cols].to_numpy(dtype=float)
        self.lo_ = np.quantile(sub, self.clip_quantile, axis=0)
        self.hi_ = np.quantile(sub, 1 - self.clip_quantile, axis=0)
        sub = np.clip(sub, self.lo_, self.hi_)
        self.mean_ = sub.mean(axis=0)
        self.std_ = np.clip(sub.std(axis=0), 1e-8, None)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns_ if c not in df.columns]
        if missing:
            raise KeyError(f"missing descriptor columns: {missing[:5]}")
        sub = df[self.columns_].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        sub = np.clip(sub, self.lo_, self.hi_)
        return (sub - self.mean_) / self.std_

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.columns_)


class TargetScaler:
    """Standardise one objective, and remember how to undo it.

    The optimiser works in standardised units so that every objective is on a
    comparable scale. Everything reported back to a human is converted here.
    """

    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=float)
        self.mean_ = float(np.nanmean(y))
        self.std_ = float(max(np.nanstd(y), 1e-8))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.std_ + self.mean_

    def inverse_std(self, s: np.ndarray) -> np.ndarray:
        """Standard deviations scale, they do not shift."""
        return np.asarray(s, dtype=float) * self.std_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)
