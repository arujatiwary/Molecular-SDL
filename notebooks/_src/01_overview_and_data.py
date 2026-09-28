# %% [markdown]
# # 1. What this project does, and getting the data
#
# ## The short version
#
# You already built a machine that picks the next experiment. It was built for
# concrete: given the mixes you have tested so far, it proposes the next batch
# to try, balancing strength, workability and cost inside the rules a mix has to
# obey.
#
# This project points that same machine at molecules. Given the compounds
# tested so far, it proposes the next batch to make, balancing how well they hit
# a drug target, how well they dissolve, and how hard they are to synthesise.
#
# The machine does not change. What changes is how you describe one recipe to it.

# %% [markdown]
# ## The one swap
#
# | Concrete | Molecule |
# | --- | --- |
# | A recipe is a list of quantities: cement, water, sand, admixture | A recipe is a SMILES string, turned into numbers by RDKit |
# | Strength, measured by crushing a cube | Potency, measured in a binding assay |
# | Workability, measured by a slump test | Solubility, measured in water |
# | Cost, calculated exactly from the recipe | Synthetic accessibility, calculated exactly from the structure |
# | Gradation limits fix where you may look | Drug-likeness filters fix where you may look |
# | Shrinkage check after predicting | Cardiac safety check after predicting |
# | Mix proportions are continuous, so you can search them with gradients | Molecules are discrete, so you score a library and pick from it |
#
# That last row is the only real structural difference, and section 5 deals
# with it.

# %% [markdown]
# ## Why anyone cares
#
# Drug discovery has the same shape as mix design. Every experiment is slow and
# expensive. Several properties matter at once and they pull against each
# other. You never have enough data. So the question is not "what is the best
# molecule", it is "given a fixed budget of experiments, where do I spend the
# next eight".
#
# A team running this kind of loop typically makes and tests a few dozen
# compounds a month. Cutting the number of rounds needed to find a good
# compromise is worth real money and real time.

# %% [markdown]
# ## The three things we trade off
#
# **Potency.** How strongly a compound hits its target. Here the target is EGFR,
# a protein that drives several lung cancers and the target of gefitinib,
# erlotinib and osimertinib. Potency is reported as pIC50: the negative log of
# the concentration that halves the target's activity. Higher is better, and
# each whole number is a tenfold improvement.
#
# **Solubility.** A compound that does not dissolve cannot be dosed. Reported
# as logS, the log of solubility in mol/L. Higher is better.
#
# **Synthesisability.** Whether a chemist can actually make it. Reported as the
# SA score, 1 for easy and 10 for hard. Lower is better, so we optimise its
# negative.
#
# ## Why these three fight each other
#
# Adding greasy bulk to a molecule usually raises potency and lowers
# solubility. Making it more elaborate often raises potency and makes it harder
# to synthesise. There is no single winner, only a set of best compromises: the
# Pareto front. Finding that front cheaply is the whole job.

# %% [markdown]
# ## The five pieces
#
# 1. **A potency model** with two tiers. A base model trained on a large,
#    loosely related dataset, plus a correction model trained on the target you
#    care about.
# 2. **A solubility model**, same two-tier shape, with a published literature
#    equation as the base tier.
# 3. **A synthesisability model**, which is not a model at all. The SA score is
#    computed exactly, the way a cost model is.
# 4. **Filters and gates.** Drug-likeness rules that shrink the search space
#    before the optimiser looks, and a cardiac safety gate applied after
#    prediction. These are two different jobs and the code keeps them apart.
# 5. **The optimiser.** qNEHVI over a library of candidate molecules, proposing
#    a batch at a time.
#
# Wrapped around all of it: conformal intervals, so the uncertainty the models
# report is uncertainty you can believe, and SHAP over substructures, so you can
# see which parts of a molecule the model is reacting to.

# %% [markdown]
# ## The data
#
# Everything here is measured data from public sources. Nothing is simulated.
#
# | Role | Dataset | Size | Source |
# | --- | --- | --- | --- |
# | Potency, base tier | KIBA kinase affinity matrix | 2,111 compounds by 229 kinases | DeepDTA repository |
# | Potency, target tier | EGFR IC50 from ChEMBL | 5,568 rows | TeachOpenCADD mirror |
# | Solubility, base tier | Delaney's published ESOL equation | prediction per compound | shipped inside the ESOL file |
# | Solubility, target tier | ESOL measured log solubility | 1,128 compounds | DeepChem |
#
# **Why KIBA is the base tier for potency.** A medicinal chemistry team starting
# an EGFR programme is not starting from nothing. They have years of data on
# other kinases, and kinase inhibitors share a lot of chemistry. KIBA is that
# prior in dataset form: one consistent affinity score for 2,111 compounds
# against 229 kinases. We remove EGFR from it before training, so the base model
# knows kinases in general and knows nothing about the target.
#
# **Upgrade path.** `molsdl.data.fetch_chembl_activities` pulls real pIC50 for
# any ChEMBL target, which lets you build the base tier in the same units as the
# target tier. Those functions were written against the documented API but were
# not run while this repo was built, so treat them as unverified until you run
# them yourself.

# %%
import sys, time, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
sys.path.insert(0, "../src")

from molsdl import chem, data, features, objectives, plotting

plotting.use_style()
RAW = "../data/raw"
PROC = Path("../data/processed"); PROC.mkdir(parents=True, exist_ok=True)
print("ready")

# %% [markdown]
# ### Download
#
# Five files, about 5 MB in total. Already-downloaded files are skipped.

# %%
paths = data.download_raw(RAW)
for k, v in paths.items():
    print(f"  {k:16s} {v.stat().st_size/1e6:6.2f} MB")

# %% [markdown]
# ### Load, and check the numbers against the files
#
# Never take a row count from a paper or a README. Count it.

# %%
kiba = data.load_kiba(RAW, exclude_uniprot="P00533")   # P00533 is EGFR
egfr = data.load_egfr(RAW)
esol_raw = pd.read_csv(f"{RAW}/esol.csv")
esol = pd.DataFrame({
    "smiles": esol_raw["smiles"].str.strip(),
    "logS": esol_raw["measured log solubility in mols per litre"].astype(float),
    # Delaney's own published estimate, which becomes the solubility base tier
    "logS_delaney": esol_raw["ESOL predicted log solubility in mols per litre"].astype(float),
})

print(f"KIBA  {kiba.shape[0]:5d} compounds, "
      f"tested against {kiba.kiba_n.min()}-{kiba.kiba_n.max()} kinases each")
print(f"EGFR  {egfr.shape[0]:5d} rows, pIC50 {egfr.pIC50.min():.2f} to {egfr.pIC50.max():.2f}")
print(f"ESOL  {esol.shape[0]:5d} rows, logS  {esol.logS.min():.2f} to {esol.logS.max():.2f}")

# %% [markdown]
# ### Standardise the structures
#
# Before any two datasets can be compared, the molecules have to be written the
# same way. The same compound can appear as a sodium salt in one source and a
# free acid in another, or drawn with a different but equivalent SMILES string.
#
# The steps: parse, keep the largest fragment (drops salts and counter-ions),
# neutralise charges, write a canonical SMILES, then compute an InChIKey. The
# InChIKey is the identity key everything downstream deduplicates on.
#
# Nothing is repaired by guessing. A structure that fails to parse is dropped
# and counted.

# %%
t0 = time.time()
kiba = chem.standardise_frame(kiba)
egfr = chem.standardise_frame(egfr)
esol = chem.standardise_frame(esol)
print(f"  {time.time()-t0:.0f}s")

# %% [markdown]
# ### Deduplicate, and look at what duplication hides
#
# This is the lesson that costs people the most. In a concrete dataset, a
# deduplication key built from mix proportions alone can collapse hundreds of
# genuinely distinct batches into one row. The molecular version is the same
# mistake wearing a different hat: two rows with different SMILES strings can be
# the same compound.
#
# Deduplicate on the InChIKey, and when the same compound has several measured
# values, keep the median and keep the spread. The spread is information: it
# tells you how noisy the assay is, which is the floor your model can never beat.

# %%
for name, df, col in [("EGFR", egfr, "pIC50"), ("ESOL", esol, "logS")]:
    dup = df.groupby("inchikey")[col].agg(["size", "min", "max"])
    multi = dup[dup["size"] > 1]
    print(f"{name}: {len(df)} rows, {df.inchikey.nunique()} unique compounds, "
          f"{len(multi)} measured more than once")
    if len(multi):
        print(f"       widest disagreement between repeats: "
              f"{(multi['max'] - multi['min']).max():.2f} units")

# %%
def collapse(df, value_col):
    g = df.groupby("inchikey").agg(
        std_smiles=("std_smiles", "first"),
        **{value_col: (value_col, "median")},
        n_measurements=(value_col, "size"),
        spread=(value_col, lambda s: float(s.max() - s.min())),
    ).reset_index()
    return g

egfr_u = collapse(egfr, "pIC50")
esol_u = esol.drop_duplicates("inchikey").reset_index(drop=True)
kiba_u = kiba.drop_duplicates("inchikey").reset_index(drop=True)
print(f"after dedup: KIBA {len(kiba_u)}, EGFR {len(egfr_u)}, ESOL {len(esol_u)}")

# %% [markdown]
# ### The leakage audit
#
# The base tier and the target tier must not be the same compounds measured
# twice, or the transfer result is an illusion. Count the overlap by InChIKey
# and report it. Here the overlap is real but small, and it is legitimate: a
# kinase team genuinely does have family data on compounds it also tested
# against its own target. Section 3 measures what happens if you remove them.

# %%
ov = set(kiba_u.inchikey) & set(egfr_u.inchikey)
print(f"compounds in both KIBA (base) and EGFR (target): {len(ov)} "
      f"({100*len(ov)/len(egfr_u):.1f}% of the EGFR set)")
print(f"compounds in both ESOL and EGFR: "
      f"{len(set(esol_u.inchikey) & set(egfr_u.inchikey))}")

# %% [markdown]
# ### Descriptors
#
# About 200 RDKit 2D descriptors per molecule: size, lipophilicity, polar
# surface area, ring counts, hydrogen bond donors and acceptors, and many
# variations on those. This is the molecular equivalent of the columns in a mix
# design table.
#
# A molecule whose descriptors fail is dropped, not filled in. Blank means not
# computed, never zero.

# %%
t0 = time.time()
D_kiba = chem.descriptors_frame(kiba_u.std_smiles, verbose=False)
D_egfr = chem.descriptors_frame(egfr_u.std_smiles, verbose=False)
D_esol = chem.descriptors_frame(esol_u.std_smiles, verbose=False)

def drop_failed(df, D, name):
    ok = D.notna().all(axis=1)
    print(f"  {name}: kept {int(ok.sum())}/{len(ok)}")
    return df[ok].reset_index(drop=True), D[ok].reset_index(drop=True)

kiba_u, D_kiba = drop_failed(kiba_u, D_kiba, "KIBA")
egfr_u, D_egfr = drop_failed(egfr_u, D_egfr, "EGFR")
esol_u, D_esol = drop_failed(esol_u, D_esol, "ESOL")
print(f"  {time.time()-t0:.0f}s, {D_kiba.shape[1]} descriptors each")

# %% [markdown]
# ### Molecular properties for the candidate library
#
# The EGFR compounds are the library the optimiser will choose from. For each
# one we compute the properties the filters and gates need, plus the SA score,
# which is one of the three objectives.

# %%
props = objectives.molecule_properties(egfr_u.std_smiles.tolist(), verbose=True)
print(props[["MW", "logP", "TPSA", "QED", "SA"]].describe().round(2).to_string())
print(f"\nPAINS hits: {int(props.PAINS.sum())} of {len(props)}")

# %% [markdown]
# ### Save everything downstream needs

# %%
pickle.dump({
    "kiba": kiba_u, "D_kiba": D_kiba,
    "egfr": egfr_u, "D_egfr": D_egfr,
    "esol": esol_u, "D_esol": D_esol,
    "props_egfr": props,
    "overlap_kiba_egfr": sorted(ov),
}, open(PROC / "datasets.pkl", "wb"))
print("wrote ../data/processed/datasets.pkl")

# %% [markdown]
# ## What just happened
#
# Four public datasets came in as text. They left as clean structures with a
# shared identity key, a descriptor matrix, and an honest count of how much they
# overlap. No value was invented, no missing number was filled in, and every
# count in this notebook was produced by counting rather than quoted.
#
# Next: turning those descriptors into something a Gaussian process can use, and
# splitting the data so the scores mean what they claim to mean.
