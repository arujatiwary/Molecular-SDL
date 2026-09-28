"""The things we optimise, and the rules a molecule has to pass.

Three roles are kept apart on purpose, because mixing them is the easiest way
to get a pipeline that looks right and is wrong:

  1. Search-space filters. Applied to the candidate pool before the optimiser
     ever sees it. Lipinski bounds, a QED floor, PAINS removal.
  2. Objectives. Potency, solubility, synthesisability. These are what the
     optimiser trades off against each other.
  3. Feasibility gates. Applied after prediction. A molecule can be predicted
     potent and soluble and still be thrown out for hERG risk.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  (RDKit ships this under Contrib)

_PAINS_PARAMS = FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_PAINS_PARAMS)


# ---------------------------------------------------------------------------
# Objective 3: synthesisability
# ---------------------------------------------------------------------------

def sa_score(smiles: str) -> float:
    """Synthetic accessibility, 1 (easy) to 10 (hard).

    Ertl and Schuffenhauer's score. It counts how common a molecule's fragments
    are in known compounds and penalises size, stereocentres and unusual rings.

    It is a heuristic, not a measurement. Because it rewards familiar fragments
    it also penalises novelty, which is worth saying out loud in any write-up.
    This is the deterministic objective, the direct counterpart of a cost model
    in a materials pipeline: no model, no uncertainty, exact for every molecule.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.nan
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Drug-likeness and filters
# ---------------------------------------------------------------------------

def molecule_properties(smiles: Sequence[str], verbose: bool = False) -> pd.DataFrame:
    """Everything the filters and gates need, in one table."""
    rows = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            rows.append({})
            continue
        rows.append({
            "MW": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "QED": QED.qed(mol),
            "SA": sa_score(s),
            "PAINS": bool(_PAINS.HasMatch(mol)),
            "basic_N": _has_basic_nitrogen(mol),
        })
        if verbose and (i + 1) % 2000 == 0:
            print(f"    properties {i + 1}/{len(smiles)}")
    return pd.DataFrame(rows)


def _has_basic_nitrogen(mol) -> bool:
    """Neutral, non-amide, non-aromatic nitrogen: the classic basic centre.

    This is the feature that shows up in most hERG pharmacophore models, which
    is why the gate below uses it.
    """
    patt = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N[!#6]);!$(N*=[O,N,S]);!$(N#*)]")
    return mol.HasSubstructMatch(patt) if patt is not None else False


def pool_filter(props: pd.DataFrame,
                mw=(150.0, 600.0), logp_max=5.5, hbd_max=5, hba_max=10,
                rotb_max=10, qed_min=0.35, drop_pains=True) -> np.ndarray:
    """Boolean mask of molecules allowed into the candidate pool.

    These are search-space constraints. They shrink where the optimiser is
    allowed to look, the same way a material standard fixes the range of a mix
    before any optimisation starts. They are not predictions and they are not
    gates.
    """
    ok = (
        props["MW"].between(*mw)
        & (props["logP"] <= logp_max)
        & (props["HBD"] <= hbd_max)
        & (props["HBA"] <= hba_max)
        & (props["rotb"] <= rotb_max)
        & (props["QED"] >= qed_min)
    )
    if drop_pains:
        ok &= ~props["PAINS"].astype(bool)
    return ok.fillna(False).to_numpy()


# ---------------------------------------------------------------------------
# Feasibility gate: hERG
# ---------------------------------------------------------------------------

def herg_risk_rule(props: pd.DataFrame, logp_cut: float = 3.7,
                   ring_cut: int = 3) -> np.ndarray:
    """Rule-based hERG risk flag.

    hERG is a cardiac potassium channel. Blocking it lengthens the QT interval
    and has killed more than one late-stage drug programme, so every medicinal
    chemistry team screens for it.

    Published pharmacophore work describes the typical blocker as a lipophilic
    molecule with a basic nitrogen flanked by aromatic rings. This function
    encodes that shape: basic nitrogen AND high logP AND several rings.

    It is a surrogate, used here so that the gate is testable with no extra
    download. Replace it with `HergModelGate` once you have measured data, and
    say in any write-up which one you used.
    """
    risky = (props["basic_N"].astype(bool)
             & (props["logP"] >= logp_cut)
             & (props["aromatic_rings"] >= ring_cut))
    return risky.fillna(False).to_numpy()


class HergModelGate:
    """Data-driven hERG gate, for when you have measured blocker labels.

    Fit a classifier on (features, is_blocker) and gate on predicted
    probability. `fit` takes the same feature matrix the GPs use, so it slots
    into the pipeline without new featurisation.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None

    def fit(self, X: np.ndarray, y_blocker: np.ndarray) -> "HergModelGate":
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(random_state=0)
        self.model.fit(X, np.asarray(y_blocker).astype(int))
        return self

    def risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit the gate first")
        return self.model.predict_proba(X)[:, 1]

    def blocks(self, X: np.ndarray) -> np.ndarray:
        return self.risk(X) >= self.threshold


def apply_gates(props: pd.DataFrame, herg_flag: np.ndarray | None = None,
                sa_max: float = 6.0) -> np.ndarray:
    """Post-prediction feasibility. True means the molecule may be proposed."""
    ok = props["SA"] <= sa_max
    ok = ok.fillna(False).to_numpy()
    if herg_flag is not None:
        ok = ok & ~np.asarray(herg_flag, dtype=bool)
    return ok
