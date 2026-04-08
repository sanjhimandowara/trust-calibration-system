import numpy as np


def compute_conflict(values, weights=None):
    """
    Measures disagreement among source values.
    Higher variance = higher conflict.
    Returns value in [0, 1].
    """
    values = np.array(values, dtype=np.float32)

    if weights is None:
        weights = np.ones_like(values, dtype=np.float32) / len(values)
    else:
        weights = np.array(weights, dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / np.sum(weights)

    weighted_mean = np.sum(values * weights)
    weighted_var = np.sum(weights * (values - weighted_mean) ** 2)

    conflict = np.sqrt(weighted_var) * 2.0
    return float(np.clip(conflict, 0.0, 1.0))


def compute_uncertainty(confidences, conflict_score):
    """
    Combines source confidence and conflict into uncertainty.
    Lower confidence + higher conflict = higher uncertainty.
    Returns value in [0, 1].
    """
    confidences = np.array(confidences, dtype=np.float32)
    avg_conf = float(np.mean(confidences))

    uncertainty = (1.0 - avg_conf) * 0.6 + conflict_score * 0.4
    return float(np.clip(uncertainty, 0.0, 1.0))


def consensus_score(values):
    """
    Measures how closely sources agree with one another.
    Higher agreement = higher consensus.
    Returns value in [0, 1].
    """
    values = np.array(values, dtype=np.float32)
    std = float(np.std(values))

    consensus = 1.0 - (std * 2.0)
    return float(np.clip(consensus, 0.0, 1.0))