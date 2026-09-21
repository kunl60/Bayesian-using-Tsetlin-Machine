import numpy as np
from causallearn.utils.cit import CIT
import logging

logger = logging.getLogger(__name__)


def _validate_xyz(X, Y, Z):
    if not isinstance(Z, list):
        if hasattr(Z, "__iter__"):
            Z = list(Z)
        else:
            raise ValueError(f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z.")

    return Z


def chi_square(X, Y, Z=[], data=None, boolean=False, significance_level=0.01, **kwargs):
    """
    Chi-square conditional independence test for discrete/categorical data.
    Tests the null hypothesis that X is independent from Y given Z.
    Returns: (None, p_value, None) if boolean=False
             True/False if boolean=True

    Delegates to causal-learn's CIT("chisq") - the same chi-square
    implementation causal-learn's own `pc()` search uses - rather than a
    from-scratch/pgmpy CI test, so the wtm_* Markov-blanket algorithms
    score independence with the identical engine used elsewhere.
    """
    if data is None:
        raise ValueError("Data must be provided")

    Z = _validate_xyz(X, Y, Z)

    columns = [X, Y] + Z
    array = data[columns].to_numpy()
    cit = CIT(array, method="chisq")
    condition_set = list(range(2, 2 + len(Z)))
    p_value = cit(0, 1, condition_set)

    if boolean:
        return p_value >= significance_level
    else:
        return None, p_value, None


def fisher_z(X, Y, Z=[], data=None, boolean=False, significance_level=0.01, **kwargs):
    if data is None:
        raise ValueError("Data must be provided")

    Z = _validate_xyz(X, Y, Z)

    n = len(data)
    dof = n - len(Z) - 3
    if dof <= 0 or data[X].std() == 0 or data[Y].std() == 0:
        # Too few samples relative to |Z|, or X/Y itself constant in this
        # sample - no reliable partial correlation to measure, so treat as
        # "can't reject independence" (see docstring).
        p_value = 1.0
    else:
        columns = [X, Y] + Z
        array = data[columns].to_numpy(dtype=float)
        cit = CIT(array, method="fisherz")
        condition_set = list(range(2, 2 + len(Z)))
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_value = cit(0, 1, condition_set)
        except ValueError:
            # Singular [X,Y,*Z] correlation submatrix - see docstring.
            p_value = 1.0

    if boolean:
        return p_value >= significance_level
    else:
        return None, p_value
