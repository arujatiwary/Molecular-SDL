# Multi-objective molecular property optimisation

A decision engine that proposes which molecules to make next, balancing how
strongly they hit a drug target, how well they dissolve, and how hard they are
to synthesise.

It is the same architecture as a self-driving-laboratory engine for concrete mix
design, pointed at molecules instead. Nothing about the decision machinery
changes. What changes is how one recipe gets described to it.

---

## The idea in plain English

Making and testing a compound takes weeks and costs money. Several properties
matter at once and they pull against each other: the changes that make a
molecule bind better usually make it dissolve worse. So the real question is
never "what is the best molecule". It is "given a budget of experiments, which
six do I run next".

This repo answers that question in a loop:

1. Fit models on everything measured so far.
2. Ask an acquisition function which six molecules to test next.
3. Test them. Here that means revealing labels that were hidden; in a lab it
   means running the assay.
4. Refit and repeat.

The result is a set of best compromises, called a Pareto front, found in fewer
experiments than picking at random or chasing one property.

## The mapping from concrete to molecules

| Concrete | Molecule |
| --- | --- |
| A recipe is a list of quantities | A recipe is a SMILES string, turned into numbers by RDKit |
| Strength, from crushing a cube | Potency, from a binding assay (pIC50) |
| Workability, from a slump test | Solubility in water (logS) |
| Cost, calculated exactly | Synthetic accessibility, calculated exactly |
| Gradation limits fix the search space | Drug-likeness filters fix the search space |
| Shrinkage check after predicting | Cardiac safety check after predicting |
| Continuous proportions, gradient search | Discrete structures, pick from a library |

That last row is the only structural difference, and notebook 5 deals with it.

## The five pieces

1. **Potency model**, two tiers: a base model trained on kinase-family data plus
   a correction model trained on the specific target.
2. **Solubility model**, same shape, with a published equation as the base tier.
3. **Synthesisability**, which is not a model. The SA score is computed exactly.
4. **Filters and gates**, kept apart on purpose: drug-likeness rules shrink the
   search space before the optimiser looks, and a cardiac safety gate is applied
   after prediction.
5. **The optimiser**: qNEHVI over a library of candidates, proposing a batch at
   a time.

Around all of it: conformal intervals so the uncertainty is trustworthy, and
SHAP over substructures so the model can say why.

---

## The data

Everything is measured public data. Nothing is simulated.

| Role | Dataset | Size after cleaning | Source |
| --- | --- | --- | --- |
| Potency, base tier | KIBA kinase affinity, EGFR column removed | 2,111 compounds against 228 kinases | DeepDTA repo |
| Potency, target tier | EGFR IC50 from ChEMBL | 5,531 unique compounds | TeachOpenCADD mirror |
| Solubility, base tier | Delaney's published ESOL equation | one value per compound | ships inside the ESOL file |
| Solubility, target tier | ESOL measured log solubility | 1,117 unique compounds | DeepChem |

EGFR drives several lung cancers and is the target of gefitinib, erlotinib and
osimertinib, so the chemistry here is real lead-optimisation chemistry.

All five raw files download automatically in notebook 1, about 5 MB total.

---

## Results

Scores are on held-out **scaffolds**, never a random split.

### Splitting honestly costs you 0.4 of R²

Same model, same features, same data. Only the split changes.

| Split | R² on EGFR potency |
| --- | --- |
| random | 0.64 |
| scaffold | 0.15 to 0.62 depending on training size |

A random split lets close analogues sit on both sides. Every number in this repo
uses a scaffold split.

### Transfer helps a lot, or not at all, and you can tell which in advance

| | solubility | potency |
| --- | --- | --- |
| base tier | Delaney's published equation | kinase family, EGFR removed |
| correlation of base with target | 0.91 | 0.26 |
| base alone, R² | 0.82 to 0.84 | about 0.02 |
| plain GP at 50 training molecules | **−0.20** | 0.11 |
| base + residual at 50 training molecules | **0.83** | 0.10 |
| base + residual at full training set | 0.86 | 0.62, same as plain |

Two findings worth reporting:

1. With an informative base tier and 50 training molecules, the two-tier model
   reaches R² 0.83 where a plain GP scores worse than guessing the mean.
2. With an uninformative base tier the two-tier model lands on top of the plain
   model. On a static test set, a weak prior costs nothing.

The diagnostic that predicts which case you are in is one line: the correlation
between base prediction and target on the training rows. Compute it before
building anything.

Then see the loop result below, which qualifies finding 2.

### Conformal intervals

90% conformal intervals on held-out solubility cover about 96% of true values,
so they are conservative rather than optimistic. See the limitations below for
what they are not yet doing.

### The closed loop

600-molecule library, 12 rounds of 6 molecules, 3 seeds, all methods starting
from the same 36 seed compounds. Hypervolume after 72 experiments, averaged over
seeds, from a starting value of 0.441:

| method | hypervolume |
| --- | --- |
| qNEHVI without the transfer tier | **0.762** |
| qNEHVI with the transfer tier | 0.702 |
| random | 0.573 |
| greedy on potency | 0.518 |

**Chasing one property finishes below random.** Greedy potency flattens after
round two. It keeps finding potent compounds that are all potent in the same
way, so the front stops growing. That is the case for multi-objective selection
in one line.

**A weak base tier is harmless on a test set and harmful in a loop.** Section 3
found the kinase-family prior changed nothing when scored on held-out
molecules. Inside the loop it consistently costs about 0.06 of hypervolume. A
static score averages over a fixed test set; a loop compounds its own choices,
so a prior that pulls predictions the wrong way sends the first batch to the
wrong part of the library and every later batch is chosen from what that taught.

The practical rule: test a base tier inside the loop before shipping it, not
only on held-out data.

Full curves in `outputs/metrics/campaign.csv` and
`outputs/figures/05_hypervolume_curves.png`.

### The model found the right chemistry on its own

SHAP over fingerprint bits ranks an aromatic nitrogen bridge and a
pyrimidine-type ring at the top. Those are the pieces of a
4-anilinoquinazoline, the scaffold behind gefitinib and erlotinib.

Turning that into a testable claim and checking it against the measured values:

| | compounds | mean pIC50 |
| --- | --- | --- |
| carry the anilino NH bridge | 3,488 | 7.16 |
| do not | 2,043 | 5.79 |

A difference of 1.37 log units, about 23-fold in potency. The model was never
told about this scaffold. It read it off the fingerprints.

---

## Layout

```
MOL/
  notebooks/           the six notebooks, in order
    _src/              the cell scripts they are built from
  src/molsdl/
    chem.py            standardisation, descriptors, fingerprints, splits
    data.py            dataset loaders, offline and ChEMBL
    features.py        descriptor pruning and scaling
    models.py          GPs, the Tanimoto kernel, the transfer model
    conformal.py       split conformal intervals
    objectives.py      SA score, filters, hERG gate
    optimise.py        qNEHVI over a pool, campaign runner
    plotting.py        one palette for every figure
  data/raw/            downloaded, gitignored
  data/processed/      built by the notebooks
  outputs/figures/     every figure
  outputs/metrics/     every table
  build_notebooks.py   rebuilds the .ipynb from notebooks/_src
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon everything installs from PyPI with no extra steps. The whole
project runs on CPU; no GPU is needed.

## Run order

Run the notebooks in order. Each one saves what the next one loads.

| Notebook | What it does | Measured |
| --- | --- | --- |
| 1 | what the project is, download, clean, deduplicate, audit overlap | 137 s |
| 2 | descriptors, fingerprints, scaffold splits | 65 s |
| 3 | the two-tier model and learning curves | 137 s |
| 4 | conformal intervals, filters, gates, building the library | 25 s |
| 5 | the closed loop against three baselines | 33 min, or 5 s from cache |
| 6 | SHAP over substructures | 140 s |

Timings are wall clock on a 2-core cloud container. Expect faster on a recent
Mac.

Notebook 5 caches its campaign to `data/processed/campaign.pkl` and reloads it
on a second run, so re-running the notebook to redraw a figure costs seconds
rather than half an hour. Set `REUSE_CACHE = False` in that notebook to force a
fresh run.

To edit a notebook, edit the matching file in `notebooks/_src/` and run
`python3 build_notebooks.py`. Editing the `.ipynb` directly works too; the
`_src` files are the readable version.

---

## What was verified, and what was not

Every number in this repo came from running the code. Nothing was quoted from a
paper or an abstract.

**Verified**: all six notebooks were executed top to bottom in a clean kernel.
Every dataset row count was counted from the downloaded file. Every model score
came from a scaffold-held-out set.

**Not verified**: `data.fetch_chembl_activities` and
`data.fetch_chembl_kinase_family` were written against the documented ChEMBL
REST API but were never run, because the machine this was built on could not
reach ebi.ac.uk. Run them, check the row count against the ChEMBL web interface
for the same target, and only then trust them.

## A trap this repo walked into and backed out of

KIBA has an EGFR column, and it looks like the obvious base tier for EGFR
potency. It is disqualified. KIBA was assembled from the same public IC50
measurements the ChEMBL EGFR set comes from, and on the 246 compounds present in
both the two agree at correlation 0.97. Using it would have produced a
spectacular transfer result that meant nothing.

The overlap audit in notebook 1 exists because of this. Run it on any new pair
of datasets before trusting a transfer number.

## Known limitations

State these in any write-up rather than hoping nobody asks.

1. **The solubility axis in the closed loop is a model output, not a
   measurement.** Potency is a real hidden label revealed on selection.
   Solubility is the conformal lower bound from a model trained on ESOL and
   applied to kinase inhibitors, which is out of its training chemistry. The
   loop demonstrates the decision machinery; the measured-data claims live in
   notebook 3.
2. **The conformal intervals are close to constant width.** With this kernel and
   this much data the GP saturates at its prior uncertainty almost everywhere,
   so the intervals are calibrated but they are not yet telling you which
   individual molecules are risky. Fixes in order of effort: more solubility
   data (AqSolDB has about 10,000 compounds), a Tanimoto kernel instead of
   descriptors, a heteroscedastic noise model.
3. **The hERG gate is a rule, not a model.** It encodes the published
   pharmacophore picture of a blocker: basic nitrogen, high logP, three or more
   aromatic rings. It flags nearly half the library, which is a real property of
   kinase inhibitors rather than a bug. `objectives.HergModelGate` takes measured
   labels and fits a classifier; use that for anything you publish.
4. **The SA score penalises novelty.** It rewards fragments that are common in
   known compounds, so a genuinely new scaffold scores as hard to make. It is a
   heuristic, not a measurement.
5. **GP fits wobble.** The plain solubility GP scored 0.64 at 400 training
   molecules and 0.50 at 781, which is an optimiser artefact rather than a real
   effect. Raise `maxiter` or the restart count if you need a stable number.
6. **The loop is small.** 600 molecules, 12 rounds, 3 seeds is enough to see a
   difference on a laptop. For anything publishable, raise the pool to a few
   thousand, the rounds to 25 and the seeds to 5.

## The upgrade path

In rough order of value per hour spent:

1. **Real pIC50 for the base tier.** Pull a kinase family from ChEMBL so base
   and target tiers are in the same units. `data.fetch_chembl_kinase_family`
   does this; it needs testing.
2. **AqSolDB for solubility.** About 10,000 measured compounds turns a
   constant-width interval into a real per-molecule assessment.
3. **A measured hERG gate.** Swap the rule for `HergModelGate`.
4. **More seeds and rounds** in the loop, so the hypervolume claim survives a
   reviewer.
5. **A latent-space search** instead of a fixed library, if you want to propose
   molecules nobody has made. This is a much bigger build with many more ways to
   fail. It is the second thing to try, not the first.

## Sources

- KIBA and Davis kinase datasets: [DeepDTA repository](https://github.com/hkmztrk/DeepDTA)
- EGFR ChEMBL compounds: [TeachOpenCADD T001](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html)
- ESOL / Delaney solubility: [DeepChem datasets](https://github.com/deepchem/deepchem)
- AqSolDB, for the upgrade path: [AqSolDB repository](https://github.com/mcsorkun/AqSolDB)
- ChEMBL REST API: [ebi.ac.uk/chembl](https://www.ebi.ac.uk/chembl/)
