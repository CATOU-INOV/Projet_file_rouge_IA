"""Détection de drift : KS-test par feature et PSI (Population Stability Index)."""
import numpy as np
from scipy import stats


def detect_data_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list = None,
    threshold: float = 0.05,
) -> dict:
    """Appliquer le test de Kolmogorov-Smirnov par feature.

    Args:
        reference: tableau de référence (n_ref, n_features) ou (n_ref,)
        current:   tableau courant   (n_cur, n_features) ou (n_cur,)
        feature_names: noms des features (optionnel)
        threshold: seuil de p-value pour déclarer le drift (défaut 0.05)

    Returns:
        dict avec, par feature : ks_statistic, p_value, is_drift
        + clés globales "any_drift" (bool) et "drift_features" (liste)
    """
    reference = np.array(reference, dtype=float)
    current = np.array(current, dtype=float)

    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results = {}
    drift_features = []

    for i, name in enumerate(feature_names):
        ref_col = reference[:, i]
        cur_col = current[:, i]

        ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
        is_drift = bool(p_value < threshold)

        results[name] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "is_drift": is_drift,
        }

        if is_drift:
            drift_features.append(name)

    results["any_drift"] = len(drift_features) > 0
    results["drift_features"] = drift_features
    results["drift_ratio"] = round(len(drift_features) / max(len(feature_names), 1), 4)

    return results


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Calculer le PSI (Population Stability Index).

    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct))

    Interprétation :
        PSI < 0.1  → stable
        0.1 <= PSI < 0.2 → changement modéré
        PSI >= 0.2 → drift significatif

    Args:
        reference: distribution de référence (1D array)
        current:   distribution courante    (1D array)
        n_bins:    nombre de bins

    Returns:
        dict avec psi, interpretation, is_drift, bins_detail
    """
    reference = np.array(reference, dtype=float).flatten()
    current = np.array(current, dtype=float).flatten()

    # Bins sur l'union des deux distributions
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    eps = 1e-10
    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi_per_bin = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi = float(np.sum(psi_per_bin))

    if psi < 0.1:
        interpretation = "stable"
    elif psi < 0.2:
        interpretation = "moderate_change"
    else:
        interpretation = "significant_drift"

    return {
        "psi": round(psi, 6),
        "interpretation": interpretation,
        "is_drift": psi >= 0.2,
        "bins_detail": {
            f"bin_{i}": {
                "ref_pct": round(float(ref_pct[i]), 4),
                "cur_pct": round(float(cur_pct[i]), 4),
                "psi_contribution": round(float(psi_per_bin[i]), 6),
            }
            for i in range(n_bins)
        },
    }
