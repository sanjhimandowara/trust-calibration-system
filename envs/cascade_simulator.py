def compute_cascade_risk(decision, conflict_score, uncertainty_score):
    """
    Estimates downstream operational risk caused by a decision.
    Higher escalation under conflict/uncertainty can create cascades.
    Returns value in [0, 1].
    """

    if decision == 0:  # IGNORE
        base_risk = 0.2
    elif decision == 1:  # INVESTIGATE
        base_risk = 0.4
    else:  # ESCALATE
        base_risk = 0.7

    risk = (
        0.5 * base_risk
        + 0.25 * conflict_score
        + 0.25 * uncertainty_score
    )

    return float(min(max(risk, 0.0), 1.0))


def cascade_penalty(decision, true_label, conflict_score, uncertainty_score):
    """
    Penalize risky wrong decisions more heavily.
    Returns penalty value (negative or zero).
    """

    risk = compute_cascade_risk(decision, conflict_score, uncertainty_score)

    if decision == true_label:
        return 0.0

    # Worst case: ignored something that should have escalated
    if decision == 0 and true_label == 2:
        return -0.8 * risk

    # Over-escalating a harmless case
    if decision == 2 and true_label == 0:
        return -0.5 * risk

    # Medium severity mismatches
    return -0.3 * risk