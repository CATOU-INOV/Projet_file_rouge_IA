"""Detection de drift. TODO: KS-test et PSI."""
import numpy as np
from scipy import stats

def detect_data_drift(reference, current, threshold=0.05):
    """TODO: KS-test par feature. Retourner dict avec ks_statistic, p_value, is_drift."""
    raise NotImplementedError("TODO")

def compute_psi(reference, current, n_bins=10):
    """TODO: PSI = sum((cur_pct - ref_pct) * ln(cur_pct/ref_pct)). Seuil: <0.1 stable, >0.2 drift."""
    raise NotImplementedError("TODO")
