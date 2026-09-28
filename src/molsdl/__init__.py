"""molsdl: a decision engine for multi-objective molecular optimisation.

The same architecture as a self-driving-laboratory engine for materials, with
molecules in place of mixes.

    chem        SMILES standardisation, descriptors, fingerprints, scaffold splits
    data        dataset loaders, offline and ChEMBL
    features    descriptor pruning and target scaling
    models      Gaussian processes, the Tanimoto kernel, the transfer model
    conformal   split conformal prediction intervals
    objectives  SA score, drug-likeness filters, the hERG gate
    optimise    qNEHVI over a candidate pool, and the campaign runner
    plotting    one palette for every figure
"""

__version__ = "0.1.0"

from . import chem, conformal, data, features, models, objectives, optimise, plotting

__all__ = ["chem", "conformal", "data", "features", "models", "objectives",
           "optimise", "plotting"]
