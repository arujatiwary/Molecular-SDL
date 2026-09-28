"""Multi-objective proposal over a fixed pool of molecules.

Why a pool and not free search. In a mix-design problem the search space is a
set of continuous proportions, so an acquisition function can be optimised with
gradients. Molecule space is discrete: there is no gradient from aspirin to
ibuprofen. The practical answer, and the one that matches how a screening
campaign actually runs, is to score a finite library and pick the best batch
from it.

qNEHVI is the acquisition function. It asks: if I run these q experiments, how
much do I expect the Pareto front to grow, given that everything I have already
measured is noisy? It handles batches and noise together, which is what an
automated lab needs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition.multi_objective import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

TDTYPE = torch.double


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=TDTYPE)


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Which rows are on the Pareto front. Assumes every column is maximised."""
    return is_non_dominated(_t(Y)).numpy()


def hypervolume(Y: np.ndarray, ref_point: np.ndarray) -> float:
    """Volume dominated by the current front, measured from the reference point.

    This is the single number that says how good a set of molecules is as a
    set. One brilliant compound with terrible solubility does not move it much;
    a spread of good compromises does.
    """
    Yt, rp = _t(Y), _t(ref_point)
    keep = (Yt > rp).all(dim=-1)
    if keep.sum() == 0:
        return 0.0
    part = DominatedPartitioning(ref_point=rp, Y=Yt[keep])
    return float(part.compute_hypervolume().item())


def reference_point(Y: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """A reference point just below the worst value seen on each objective.

    Fix this once at the start of a run and never change it. A moving reference
    point makes hypervolume curves impossible to compare.
    """
    Y = np.asarray(Y, dtype=float)
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    return lo - margin * np.clip(hi - lo, 1e-9, None)


def build_model_list(models: list) -> ModelListGP:
    """Wrap one GP per objective into a single multi-output model."""
    return ModelListGP(*models)


def known_objective_model(col: int) -> GenericDeterministicModel:
    """A model for an objective that is already known for every candidate.

    Synthetic accessibility is computed exactly, so fitting a GP to it would be
    theatre. This reads the value straight off a column of the feature row, the
    same way a cost model reads a price list. The objective still takes part in
    the hypervolume calculation, it just carries no uncertainty.
    """
    return GenericDeterministicModel(lambda Z, c=col: Z[..., c].unsqueeze(-1))


def _list_sampler(n_models: int, n_samples: int, n_gp: int = 1) -> ListSampler:
    """Monte Carlo sampler matched to a mixed list of GP and known objectives."""
    shape = torch.Size([n_samples])
    parts = [IIDNormalSampler(shape) for _ in range(n_gp)]
    parts += [IndexSampler(shape) for _ in range(n_models - n_gp)]
    return ListSampler(*parts)


def propose_batch(model_list, X_pool: np.ndarray,
                  X_observed: np.ndarray, ref_point: np.ndarray, q: int = 8,
                  max_batch_size: int = 256, seed: int | None = None,
                  n_samples: int = 32, n_gp: int = 1) -> np.ndarray:
    """Pick q molecules from the pool. Returns row indices into X_pool.

    X_pool must be the augmented feature matrix for the *available* candidates
    only, so anything already measured or blocked by a gate should be removed
    before calling this.

    `n_samples` is the number of Monte Carlo posterior samples. The BoTorch
    default is 128; 32 runs about twice as fast and, on a discrete pool of a
    few hundred, picks nearly the same batch. Raise it if your batches look
    unstable across seeds.
    """
    if seed is not None:
        torch.manual_seed(seed)
    choices = _t(X_pool)
    n_out = int(model_list.num_outputs)
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=_t(ref_point).tolist(),
        X_baseline=_t(X_observed),
        prune_baseline=True,
        cache_root=False,          # our posterior is composed, so no root cache
        sampler=_list_sampler(n_out, n_samples, n_gp=n_gp),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cand, _ = optimize_acqf_discrete(
            acq, q=q, choices=choices, max_batch_size=max_batch_size,
            unique=True,
        )
    # map the returned rows back to pool indices
    idx = []
    cand_np = cand.detach().numpy()
    for row in cand_np:
        d = np.abs(X_pool - row).sum(axis=1)
        idx.append(int(np.argmin(d)))
    return np.array(idx)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def propose_random(available: np.ndarray, q: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(available, size=min(q, len(available)), replace=False)


def propose_greedy(scores: np.ndarray, available: np.ndarray, q: int) -> np.ndarray:
    """Take the q highest predicted values of one objective. The obvious thing
    to do, and the thing that fills your Pareto front with one kind of molecule.
    """
    order = available[np.argsort(-scores[available])]
    return order[:q]


@dataclass
class LoopResult:
    """What one optimisation run produced."""
    method: str
    seed: int
    hv: np.ndarray                 # hypervolume after each round
    n_measured: np.ndarray         # cumulative experiments used
    selected: list                 # pool indices chosen, per round
    pareto_idx: np.ndarray         # final Pareto set, as pool indices

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({
            "method": self.method, "seed": self.seed,
            "round": np.arange(len(self.hv)),
            "n_measured": self.n_measured, "hypervolume": self.hv,
        })


def run_campaign(method: str, X_pool: np.ndarray, Y_true: np.ndarray,
                 ref_point: np.ndarray, init_idx: np.ndarray,
                 fit_fn=None, known_cols: tuple[int, ...] = (),
                 n_rounds: int = 12, q: int = 6, seed: int = 0,
                 allowed: np.ndarray | None = None, n_samples: int = 32,
                 max_batch_size: int = 256, verbose: bool = True) -> LoopResult:
    """Run one simulated campaign.

    Each round: refit the surrogate on what has been measured, propose q
    molecules, reveal their true values, record the hypervolume. Revealing a
    label is what stands in for running the assay.

    method is one of 'qNEHVI', 'random', 'greedy potency'.
    fit_fn(observed_indices) must return a fitted model for the measured
    objective, and is only needed for qNEHVI and greedy.
    known_cols gives the columns of X_pool that carry objectives already known
    for every candidate.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(X_pool)
    pool_ids = np.arange(n_pool)
    allowed = pool_ids if allowed is None else np.asarray(allowed)

    measured = list(init_idx)
    hv_hist, n_hist, chosen = [], [], []
    hv_hist.append(hypervolume(Y_true[measured], ref_point))
    n_hist.append(len(measured))

    for r in range(n_rounds):
        avail = np.setdiff1d(allowed, measured)
        if len(avail) == 0:
            break
        if method == "random":
            pick = propose_random(avail, q, rng)
        elif method == "greedy potency":
            model = fit_fn(np.array(measured))
            from .models import gp_predict
            mu, _ = gp_predict(model, X_pool)
            pick = propose_greedy(mu, avail, q)
        else:
            model = fit_fn(np.array(measured))
            ml = ModelList(model, *[known_objective_model(c) for c in known_cols])
            local = propose_batch(ml, X_pool[avail], X_pool[np.array(measured)],
                                  ref_point, q=q, seed=seed + r,
                                  n_samples=n_samples,
                                  max_batch_size=max_batch_size)
            pick = avail[local]
        measured.extend(int(i) for i in pick)
        chosen.append(np.asarray(pick))
        hv_hist.append(hypervolume(Y_true[measured], ref_point))
        n_hist.append(len(measured))
        if verbose:
            print(f"    {method} seed {seed} round {r+1:2d}: "
                  f"{len(measured):4d} measured, hypervolume {hv_hist[-1]:.3f}",
                  flush=True)

    Ym = Y_true[measured]
    front_local = pareto_mask(Ym)
    return LoopResult(method=method, seed=seed, hv=np.array(hv_hist),
                      n_measured=np.array(n_hist), selected=chosen,
                      pareto_idx=np.array(measured)[front_local])


def top_fraction_found(Y_true: np.ndarray, measured: np.ndarray,
                       ref_point: np.ndarray, frac: float = 0.01) -> float:
    """Share of the true best molecules the run has actually measured.

    'Best' here means on the Pareto front of the full labelled pool, so this
    only works in a simulated campaign where every label is known in advance.
    """
    front = pareto_mask(Y_true)
    front_idx = np.where(front)[0]
    if len(front_idx) == 0:
        return 0.0
    hit = np.intersect1d(front_idx, measured)
    return len(hit) / len(front_idx)
