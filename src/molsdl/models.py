"""Gaussian process surrogates, including the base-plus-residual transfer model.

The transfer idea in one sentence: a model trained on a large, loosely related
dataset supplies a prior mean, and a second model trained on the small dataset
you actually care about learns the correction.

    final prediction  = base prediction + learned correction
    final variance    = base variance   + correction variance

How that is implemented here: the base model's prediction and standard
deviation are computed once and carried as two extra columns on every feature
row. The residual GP uses a mean function that simply reads the base column,
and a kernel that ignores those two columns. That keeps the maths identical to
the formula above while staying a plain BoTorch SingleTaskGP, so the
acquisition function, model lists and fitting utilities all work unchanged.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

TDTYPE = torch.double


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=TDTYPE)


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto (Jaccard) similarity as a GP kernel.

    For binary or count fingerprints,

        k(x, y) = <x, y> / (|x|^2 + |y|^2 - <x, y>)

    This is the similarity measure chemists already use to decide whether two
    molecules are alike, so it encodes real chemical knowledge that a distance
    on descriptor columns does not. It also has almost no hyperparameters,
    which makes it fast and stable to fit on small training sets.

    Wrap it in a ScaleKernel to give it an output scale.
    """

    is_stationary = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch is not supported")
        x1s = (x1 ** 2).sum(-1, keepdim=True)
        x2s = (x2 ** 2).sum(-1, keepdim=True)
        if diag:
            dot = (x1 * x2).sum(-1)
            denom = x1s.squeeze(-1) + x2s.squeeze(-1) - dot
            return dot / denom.clamp_min(1e-9)
        dot = x1 @ x2.transpose(-2, -1)
        denom = x1s + x2s.transpose(-2, -1) - dot
        return dot / denom.clamp_min(1e-9)


class ColumnMean(gpytorch.means.Mean):
    """A prior mean that reads a fixed column of the input.

    That column holds the base model's prediction, so the GP starts from the
    base model and learns only what the base model got wrong.
    """

    def __init__(self, col: int):
        super().__init__()
        self.col = int(col)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.col]


class TransferGP(SingleTaskGP):
    """SingleTaskGP whose prior mean is a base model's prediction.

    Set `base_sd_col` to also add the base model's variance to the predictive
    variance, which is the 'variances add' rule. Leave it None to keep only the
    residual GP's own uncertainty.
    """

    def __init__(self, train_X, train_Y, train_Yvar=None, base_mean_col: int | None = None,
                 n_kernel_dims: int | None = None, base_sd_col: int | None = None,
                 kernel: str = "matern"):
        d = train_X.shape[-1]
        n_kernel_dims = d if n_kernel_dims is None else n_kernel_dims
        dims = tuple(range(n_kernel_dims))
        if kernel == "tanimoto":
            covar = ScaleKernel(TanimotoKernel(active_dims=dims))
        else:
            covar = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=n_kernel_dims, active_dims=dims)
            )
        mean = ColumnMean(base_mean_col) if base_mean_col is not None else None
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar,
                         covar_module=covar, mean_module=mean, outcome_transform=None)
        self.kernel_name = kernel
        self._base_sd_col = base_sd_col

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None, **kwargs):
        post = super().posterior(X, output_indices=output_indices,
                                 observation_noise=observation_noise,
                                 posterior_transform=None, **kwargs)
        if self._base_sd_col is not None:
            extra = X[..., self._base_sd_col].clamp_min(0.0) ** 2
            mvn = post.distribution
            eye = torch.diag_embed(extra)
            new = gpytorch.distributions.MultivariateNormal(
                mvn.mean, mvn.lazy_covariance_matrix.add_jitter(0.0).to_dense() + eye
            )
            from botorch.posteriors.gpytorch import GPyTorchPosterior
            post = GPyTorchPosterior(new)
        if posterior_transform is not None:
            return posterior_transform(post)
        return post


def fit_gp(X: np.ndarray, y: np.ndarray, yvar: np.ndarray | None = None,
           base_mean_col: int | None = None, n_kernel_dims: int | None = None,
           base_sd_col: int | None = None, verbose: bool = False,
           maxiter: int = 100, kernel: str = "matern",
           max_attempts: int = 3) -> TransferGP:
    """Fit one GP. Targets must already be standardised.

    `maxiter` caps the L-BFGS steps. Left uncapped, the optimiser can spend
    many minutes on a few thousand molecules for a fit that is no better; 60 to
    100 steps gets the same answer in seconds. Raise it if you see the
    lengthscales still moving.
    """
    tx = _t(X)
    ty = _t(y).reshape(-1, 1)
    tv = None if yvar is None else _t(yvar).reshape(-1, 1).clamp_min(1e-6)
    model = TransferGP(tx, ty, train_Yvar=tv, base_mean_col=base_mean_col,
                       n_kernel_dims=n_kernel_dims, base_sd_col=base_sd_col,
                       kernel=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}},
                             max_attempts=max_attempts)
        except Exception as exc:
            # A failed fit leaves the model at its priors, which is usable but
            # poor. Say so rather than letting it pass silently.
            print(f"    warning: GP fit did not converge ({type(exc).__name__}); "
                  f"using prior hyperparameters")
    model.eval()
    if verbose and kernel != "tanimoto":
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        print(f"    fitted GP on {len(y)} points, "
              f"lengthscale median {np.median(ls):.2f}")
    return model


@torch.no_grad()
def gp_predict(model: SingleTaskGP, X: np.ndarray, observation_noise: bool = False
               ) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation, in standardised units."""
    model.eval()
    post = model.posterior(_t(X), observation_noise=observation_noise)
    mu = post.mean.squeeze(-1).detach().numpy()
    sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).detach().numpy()
    return mu, sd


@dataclass
class TransferSurrogate:
    """The full two-tier model for one objective.

    Holds the base GP, the residual GP, and the bookkeeping needed to turn
    standardised numbers back into real units.
    """
    name: str
    base_model: SingleTaskGP | None
    residual_model: SingleTaskGP | None = None
    base_scaler: object = None          # TargetScaler for the base property
    target_scaler: object = None        # TargetScaler for the target property
    base_mean_col: int | None = None
    base_sd_col: int | None = None
    n_feat: int = 0
    notes: dict = field(default_factory=dict)

    def augment(self, X_feat: np.ndarray) -> np.ndarray:
        """Attach the base prediction and its standard deviation as two columns."""
        if self.base_model is None:
            return np.asarray(X_feat, dtype=float)
        mu, sd = gp_predict(self.base_model, X_feat)
        return np.column_stack([np.asarray(X_feat, dtype=float), mu, sd])

    def predict(self, X_aug: np.ndarray, raw_units: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the target property. X_aug must come from `augment`."""
        mu, sd = gp_predict(self.residual_model, X_aug)
        if raw_units and self.target_scaler is not None:
            return self.target_scaler.inverse(mu), self.target_scaler.inverse_std(sd)
        return mu, sd

    def predict_base_only(self, X_aug: np.ndarray, raw_units: bool = True
                          ) -> tuple[np.ndarray, np.ndarray]:
        """What the base model alone says, on the target property's scale.

        The base model predicts a different property, so its output is mapped
        onto the target scale by the affine calibration stored in `notes`.
        """
        mu = X_aug[:, self.base_mean_col]
        sd = X_aug[:, self.base_sd_col]
        if raw_units and self.target_scaler is not None:
            return self.target_scaler.inverse(mu), self.target_scaler.inverse_std(sd)
        return mu, sd


def calibrate_base(base_pred: np.ndarray, y_target: np.ndarray) -> tuple[float, float]:
    """Least squares fit of base output onto the target scale.

    The base tier and the target tier often measure related but differently
    scaled things. One straight line puts the base prediction in the target's
    units before the residual GP sees it. Fitted on training rows only.
    """
    A = np.column_stack([base_pred, np.ones_like(base_pred)])
    coef, *_ = np.linalg.lstsq(A, y_target, rcond=None)
    return float(coef[0]), float(coef[1])


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "R2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "MAE": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
    }


def bootstrap_beats(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    n_boot: int = 2000, seed: int = 0) -> float:
    """Probability that model A has lower RMSE than model B, by bootstrap.

    Reports how often A wins across resamples of the test set, so a small win
    on one split does not get read as a real win.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    wins = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = np.sqrt(np.mean((y_true[idx] - pred_a[idx]) ** 2))
        rb = np.sqrt(np.mean((y_true[idx] - pred_b[idx]) ** 2))
        wins += int(ra < rb)
    return wins / n_boot
