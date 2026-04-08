import numpy as np
from models.schemas import DECISIONS


def generate_explanation(
    decision,
    values,
    confidences,
    weights,
    conflict,
    uncertainty
):
    """
    Generates a human-readable explanation of the decision.
    """

    values = np.array(values)
    confidences = np.array(confidences)
    weights = np.array(weights)

    top_source = int(np.argmax(weights))
    top_value = float(values[top_source])
    top_conf = float(confidences[top_source])

    explanation = {
        "decision": DECISIONS[decision],
        "top_source_index": top_source,
        "top_source_value": round(top_value, 3),
        "top_source_confidence": round(top_conf, 3),
        "conflict": round(float(conflict), 3),
        "uncertainty": round(float(uncertainty), 3),
    }

    return explanation