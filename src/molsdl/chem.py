"""Molecule handling: standardisation, descriptors, fingerprints, scaffold splits.

Everything that turns a SMILES string into numbers lives here. Nothing in this
file knows about Gaussian processes or optimisation.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# One generator reused everywhere. Radius 2, 2048 bits, count-based.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Standardiser pieces, built once because they are expensive to construct.
_LFC = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()
_TE = rdMolStandardize.TautomerEnumerator()
# Without these caps the enumerator can spend minutes on one large molecule.
_TE.SetMaxTautomers(64)
_TE.SetMaxTransforms(64)


def standardise_smiles(smiles: str, canonical_tautomer: bool = False) -> str | None:
    """Clean one SMILES string.

    Steps: parse, strip salts and counter-ions, neutralise charges, optionally
    canonicalise the tautomer, then write a canonical SMILES back out.

    Tautomer canonicalisation is off by default. It is slow on large molecules
    and it can pick a form that disagrees with how the compound was recorded.
    Turn it on if your sources disagree about tautomers.

    Returns None when the string does not parse. We never guess a repair.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LFC.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        if canonical_tautomer:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mol = _TE.Canonicalize(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def inchikey(smiles: str) -> str | None:
    """InChIKey for a standardised SMILES. This is the identity key we dedup on."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def standardise_frame(
    df: pd.DataFrame, smiles_col: str = "smiles", verbose: bool = True,
    canonical_tautomer: bool = False
) -> pd.DataFrame:
    """Standardise a whole table and attach std_smiles and inchikey columns.

    Rows that fail to parse are dropped, and the count is reported. We never
    impute a molecule.
    """
    out = df.copy()
    out["std_smiles"] = [standardise_smiles(s, canonical_tautomer)
                         for s in out[smiles_col]]
    n_before = len(out)
    out = out[out["std_smiles"].notna()].copy()
    out["inchikey"] = [inchikey(s) for s in out["std_smiles"]]
    out = out[out["inchikey"].notna()].copy()
    if verbose:
        print(f"  standardised {len(out)}/{n_before} rows "
              f"({n_before - len(out)} failed to parse)")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

# Descriptors that are constant, near-constant, or that blow up numerically on
# ordinary drug-like molecules. Dropping them here keeps the GP well behaved.
_DESCRIPTOR_BLOCKLIST = {"Ipc", "MaxPartialCharge", "MinPartialCharge",
                         "MaxAbsPartialCharge", "MinAbsPartialCharge",
                         "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI",
                         "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW",
                         "BCUT2D_MRHI", "BCUT2D_MRLOW"}

DESCRIPTOR_NAMES: list[str] = [
    name for name, _ in Descriptors._descList if name not in _DESCRIPTOR_BLOCKLIST
]


def descriptors_frame(smiles: Sequence[str], verbose: bool = True) -> pd.DataFrame:
    """RDKit 2D descriptors for a list of standardised SMILES.

    Returns a DataFrame with one row per input and one column per descriptor.
    A molecule that fails gives a row of NaN. We leave the NaN in place rather
    than filling it, and drop those rows later if they exist.
    """
    rows = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            rows.append({n: np.nan for n in DESCRIPTOR_NAMES})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                d = Descriptors.CalcMolDescriptors(mol)
            except Exception:
                d = {}
        rows.append({n: d.get(n, np.nan) for n in DESCRIPTOR_NAMES})
        if verbose and (i + 1) % 2000 == 0:
            print(f"    descriptors {i + 1}/{len(smiles)}")
    out = pd.DataFrame(rows, columns=DESCRIPTOR_NAMES)
    return out.replace([np.inf, -np.inf], np.nan)


def morgan_counts(smiles: Sequence[str]) -> np.ndarray:
    """Morgan count fingerprint, radius 2, 2048 bits.

    Used only for the LightGBM/SHAP branch. The Gaussian process uses
    descriptors, because a GP with an RBF kernel on raw bits behaves poorly.
    """
    X = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if mol is None:
            continue
        fp = _MORGAN.GetCountFingerprint(mol)
        for bit, count in fp.GetNonzeroElements().items():
            X[i, bit] = count
    return X


def morgan_bit_info(smiles: str) -> dict[int, tuple]:
    """Map each set bit back to the (atom index, radius) environments it came from.

    This is what lets a SHAP value on bit 1234 be drawn as an actual substructure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    _MORGAN.GetCountFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def bit_to_smarts(smiles: str, bit: int) -> str | None:
    """Write the substructure behind one Morgan bit as a SMILES fragment."""
    mol = Chem.MolFromSmiles(smiles)
    info = morgan_bit_info(smiles)
    if mol is None or bit not in info or not info[bit]:
        return None
    atom_idx, radius = info[bit][0]
    if radius == 0:
        return mol.GetAtomWithIdx(int(atom_idx)).GetSymbol()
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_idx))
    amap: dict = {}
    sub = Chem.PathToSubmol(mol, env, atomMap=amap)
    try:
        return Chem.MolToSmiles(sub)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold: the ring systems plus the linkers between them."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_split(
    smiles: Sequence[str],
    frac_train: float = 0.7,
    frac_valid: float = 0.15,
    seed: int = 0,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split molecules so that no scaffold appears on both sides.

    A random split lets close analogues sit in train and test at once, which
    inflates every score you report. Scaffold splitting is the honest default
    for molecular data.

    balanced=True places the largest scaffold groups in train first, then fills
    valid and test with the rest. This is the DeepChem convention.
    """
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)

    groups = list(scaffolds.values())
    if balanced:
        groups.sort(key=len, reverse=True)
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)

    n = len(smiles)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    train, valid, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def scaffold_folds(smiles: Sequence[str], n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each molecule a fold id, keeping whole scaffold groups together.

    Use this for leave-one-scaffold-group-out style cross validation.
    """
    scaffolds: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        scaffolds.setdefault(murcko_scaffold(s), []).append(i)
    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    fold_sizes = np.zeros(n_folds, dtype=int)
    fold_of = np.empty(len(smiles), dtype=int)
    for g in groups:
        # greedy: put the next group in the currently smallest fold
        k = int(np.argmin(fold_sizes + rng.random(n_folds) * 1e-6))
        for i in g:
            fold_of[i] = k
        fold_sizes[k] += len(g)
    return fold_of


def tanimoto_matrix(fps_a: np.ndarray, fps_b: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between two sets of binary fingerprints."""
    a = (fps_a > 0).astype(np.float32)
    b = (fps_b > 0).astype(np.float32)
    inter = a @ b.T
    na = a.sum(1)[:, None]
    nb = b.sum(1)[None, :]
    return inter / np.clip(na + nb - inter, 1e-9, None)


def maxmin_pick(fps: np.ndarray, n_pick: int, seed: int = 0) -> np.ndarray:
    """Pick a diverse starting set with the MaxMin algorithm.

    This is the molecular version of spreading your first experiments across
    the design space instead of clustering them in one corner.
    """
    rng = np.random.default_rng(seed)
    n = fps.shape[0]
    n_pick = min(n_pick, n)
    picked = [int(rng.integers(n))]
    sim_to_picked = tanimoto_matrix(fps, fps[picked]).max(axis=1)
    while len(picked) < n_pick:
        sim_to_picked[picked] = 2.0  # never repick
        nxt = int(np.argmin(sim_to_picked))
        picked.append(nxt)
        new_sim = tanimoto_matrix(fps, fps[[nxt]]).ravel()
        sim_to_picked = np.maximum(sim_to_picked, new_sim)
    return np.array(picked)
