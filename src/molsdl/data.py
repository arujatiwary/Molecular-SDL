"""Dataset loaders.

Two routes are supported, and both use real measured data.

OFFLINE route (the default, and the one every number in this repo was checked
against):
  potency base tier ...... KIBA kinase affinity matrix, 2,111 compounds by 229
                           kinases, from the DeepDTA repository
  potency target tier .... EGFR IC50 from ChEMBL, 5,568 compounds, via the
                           TeachOpenCADD mirror
  solubility ............. ESOL / Delaney, 1,128 compounds, via DeepChem

ONLINE route (richer, for when you want the full kinase family in real pIC50
units): pulls straight from the ChEMBL REST API. The functions are written
against the documented API but were NOT executed while this repo was built,
because the machine it was built on could not reach ebi.ac.uk. Treat them as
unverified until you run them.
"""
from __future__ import annotations

import json
import pickle
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_URLS = {
    "esol": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "egfr_chembl": "https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T001_query_chembl/data/EGFR_compounds.csv",
    "kiba_ligands": "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/ligands_can.txt",
    "kiba_proteins": "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/proteins.txt",
    "kiba_Y": "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/Y",
}

# UniProt accessions for a few kinases worth targeting.
UNIPROT = {"EGFR": "P00533", "ABL1": "P00519", "VEGFR2": "P35968", "BRAF": "P15056"}


def _raw_dir(root: str | Path = "../data/raw") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ssl_context() -> "ssl.SSLContext":
    """An SSL context that works on a fresh python.org install on macOS.

    Python downloaded from python.org ships its own OpenSSL and does not use
    the system keychain, so until you run its 'Install Certificates.command'
    every HTTPS request fails with CERTIFICATE_VERIFY_FAILED. Pointing at
    certifi's bundle fixes it without touching the install.

    Certificates are still verified. Nothing here disables that.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_raw(root: str | Path = "../data/raw", force: bool = False,
                 timeout: int = 120) -> dict[str, Path]:
    """Fetch the five raw files. Skips anything already on disk."""
    d = _raw_dir(root)
    names = {"esol": "esol.csv", "egfr_chembl": "egfr_chembl.csv",
             "kiba_ligands": "kiba_ligands.txt", "kiba_proteins": "kiba_proteins.txt",
             "kiba_Y": "kiba_Y"}
    ctx = _ssl_context()
    paths = {}
    for key, fname in names.items():
        dest = d / fname
        if force or not dest.exists():
            print(f"  downloading {fname} ...")
            try:
                with urllib.request.urlopen(RAW_URLS[key], timeout=timeout,
                                            context=ctx) as fh:
                    payload = fh.read()
            except urllib.error.URLError as exc:
                if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                    raise RuntimeError(
                        "HTTPS certificate verification failed. On macOS with a "
                        "python.org build, run the installer's certificate script:\n"
                        "  /Applications/Python\\ 3.12/Install\\ Certificates.command\n"
                        "or install certifi (pip install certifi) and rerun this cell."
                    ) from exc
                raise
            dest.write_bytes(payload)
        paths[key] = dest
    return paths


# ---------------------------------------------------------------------------
# Offline loaders
# ---------------------------------------------------------------------------

def load_esol(root: str | Path = "../data/raw") -> pd.DataFrame:
    """ESOL: 1,128 compounds with measured log solubility in mol/L.

    Columns returned: smiles, logS, name.
    """
    df = pd.read_csv(_raw_dir(root) / "esol.csv")
    out = pd.DataFrame({
        "smiles": df["smiles"].str.strip(),
        "logS": df["measured log solubility in mols per litre"].astype(float),
        "name": df["Compound ID"],
    })
    return out.reset_index(drop=True)


def load_egfr(root: str | Path = "../data/raw", min_pic50: float | None = None) -> pd.DataFrame:
    """EGFR potency from ChEMBL: 5,568 compounds with pIC50.

    EGFR is the target behind gefitinib, erlotinib and osimertinib, so the
    chemistry here is real lead optimisation chemistry, not benchmark filler.

    Columns returned: smiles, pIC50, chembl_id.
    """
    df = pd.read_csv(_raw_dir(root) / "egfr_chembl.csv", index_col=0)
    out = pd.DataFrame({
        "smiles": df["smiles"].str.strip(),
        "pIC50": df["pIC50"].astype(float),
        "chembl_id": df["molecule_chembl_id"],
    })
    if min_pic50 is not None:
        out = out[out["pIC50"] >= min_pic50]
    return out.reset_index(drop=True)


def load_kiba(root: str | Path = "../data/raw", exclude_uniprot: str | None = None
              ) -> pd.DataFrame:
    """KIBA kinase affinity, summarised per compound.

    KIBA folds IC50, Ki and Kd measurements across many kinase assays into one
    score. Higher means stronger binding. We reduce the 2,111 by 229 matrix to
    a few per-compound numbers that describe how a compound behaves against
    kinases in general:

      kiba_max   strongest affinity seen against any kinase
      kiba_mean  average affinity across the kinases it was tested on
      kiba_n     how many kinases it was tested against

    Pass exclude_uniprot to drop one target from the summary. That is what
    makes the base tier honest: if you are going to predict EGFR, the base
    model must not have seen EGFR.
    """
    d = _raw_dir(root)
    ligands = json.load(open(d / "kiba_ligands.txt"))
    proteins = json.load(open(d / "kiba_proteins.txt"))
    with open(d / "kiba_Y", "rb") as fh:
        Y = np.asarray(pickle.load(fh, encoding="latin1"), dtype=float)

    keys = list(proteins.keys())
    keep = np.ones(len(keys), dtype=bool)
    if exclude_uniprot is not None:
        if exclude_uniprot in proteins:
            keep[keys.index(exclude_uniprot)] = False
        else:
            print(f"  note: {exclude_uniprot} not in KIBA, nothing excluded")
    Ysub = Y[:, keep]

    with np.errstate(invalid="ignore"):
        kmax = np.nanmax(np.where(np.isfinite(Ysub), Ysub, np.nan), axis=1)
        kmean = np.nanmean(np.where(np.isfinite(Ysub), Ysub, np.nan), axis=1)
    kn = np.isfinite(Ysub).sum(axis=1)

    out = pd.DataFrame({
        "chembl_id": list(ligands.keys()),
        "smiles": list(ligands.values()),
        "kiba_max": kmax,
        "kiba_mean": kmean,
        "kiba_n": kn,
    })
    out = out[out["kiba_n"] > 0].reset_index(drop=True)
    return out


def load_kiba_target(root: str | Path = "../data/raw", uniprot: str = "P00533"
                     ) -> pd.DataFrame:
    """KIBA scores for one kinase only. Useful as a second opinion on a target."""
    d = _raw_dir(root)
    ligands = json.load(open(d / "kiba_ligands.txt"))
    proteins = json.load(open(d / "kiba_proteins.txt"))
    with open(d / "kiba_Y", "rb") as fh:
        Y = np.asarray(pickle.load(fh, encoding="latin1"), dtype=float)
    keys = list(proteins.keys())
    if uniprot not in proteins:
        raise KeyError(f"{uniprot} not present in KIBA")
    col = Y[:, keys.index(uniprot)]
    m = np.isfinite(col)
    return pd.DataFrame({
        "chembl_id": np.array(list(ligands.keys()))[m],
        "smiles": np.array(list(ligands.values()))[m],
        "kiba": col[m],
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Online loaders (ChEMBL REST). Not executed when this repo was built.
# ---------------------------------------------------------------------------

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_chembl_activities(target_chembl_id: str, standard_type: str = "IC50",
                            page_size: int = 1000, max_pages: int = 200
                            ) -> pd.DataFrame:
    """Pull activities for one ChEMBL target.

    UNVERIFIED: written against the documented ChEMBL REST API but not run
    during development. Run it, check the row count against the ChEMBL web
    interface for the same target, and only then trust it.

    Keeps exact values in nM, drops censored values (those with a relation of
    '>' or '<'), and converts to pIC50.
    """
    rows, offset = [], 0
    for _ in range(max_pages):
        url = (f"{CHEMBL_BASE}/activity.json?target_chembl_id={target_chembl_id}"
               f"&standard_type={standard_type}&limit={page_size}&offset={offset}")
        with urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as fh:
            payload = json.load(fh)
        batch = payload.get("activities", [])
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        if payload.get("page_meta", {}).get("next") is None:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["standard_units"] == "nM"]
    df = df[df["standard_relation"] == "="]           # drop censored values
    df = df[df["standard_value"].notna()]
    df["standard_value"] = df["standard_value"].astype(float)
    df = df[df["standard_value"] > 0]
    df["pIC50"] = 9.0 - np.log10(df["standard_value"])
    df = df.rename(columns={"canonical_smiles": "smiles",
                            "molecule_chembl_id": "chembl_id"})
    # one value per molecule: median across assays, and keep the spread
    g = df.groupby("chembl_id").agg(
        smiles=("smiles", "first"),
        pIC50=("pIC50", "median"),
        n_measurements=("pIC50", "size"),
        pIC50_spread=("pIC50", lambda s: float(s.max() - s.min())),
    ).reset_index()
    return g


def fetch_chembl_kinase_family(family_targets: list[str], standard_type: str = "IC50"
                               ) -> pd.DataFrame:
    """Base tier for potency: pool activities across many kinase targets.

    UNVERIFIED, same caveat as above. Returns one row per compound with the
    median pIC50 across the family, which is the family-level prior the base
    GP learns.
    """
    frames = []
    for t in family_targets:
        print(f"  fetching {t} ...")
        f = fetch_chembl_activities(t, standard_type=standard_type)
        if not f.empty:
            f["target"] = t
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    return allf.groupby("chembl_id").agg(
        smiles=("smiles", "first"),
        pIC50_family=("pIC50", "median"),
        n_targets=("target", "nunique"),
    ).reset_index()