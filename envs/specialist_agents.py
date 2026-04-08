import numpy as np


def anomaly_specialist(values, confidences):
    """
    Specialist focused more on anomaly detector signal.
    """
    return 0.7 * values[0] + 0.3 * confidences[0]


def network_specialist(values, confidences):
    """
    Specialist focused more on network monitor signal.
    """
    return 0.7 * values[1] + 0.3 * confidences[1]


def rule_specialist(values, confidences):
    """
    Specialist focused more on rule engine signal.
    """
    return 0.75 * values[2] + 0.25 * confidences[2]


def get_specialist_scores(values, confidences):
    """
    Returns scores from all three specialists.
    """
    values = np.array(values, dtype=np.float32)
    confidences = np.array(confidences, dtype=np.float32)

    scores = np.array([
        anomaly_specialist(values, confidences),
        network_specialist(values, confidences),
        rule_specialist(values, confidences)
    ], dtype=np.float32)

    return np.clip(scores, 0.0, 1.0)


def compute_ensemble_assessment(values, confidences):
    """
    Average of specialist opinions.
    """
    scores = get_specialist_scores(values, confidences)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def compute_specialist_agreement(values, confidences):
    """
    Measures how much specialists agree with each other.
    Higher agreement = closer scores.
    """
    scores = get_specialist_scores(values, confidences)
    std = float(np.std(scores))

    agreement = 1.0 - (std * 2.0)
    return float(np.clip(agreement, 0.0, 1.0))