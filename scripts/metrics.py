"""Shared per-match scoring for the ATP probability comparisons."""

from math import fsum, inf, log, log1p


def score_probabilities(outcomes, predictions):
    """Align (match_id, a_won) and (match_id, p_a) pairs from a single split."""
    by_match = {}
    for match_id, value in predictions:
        if match_id in by_match:
            raise ValueError(f"Duplicate prediction for {match_id}")
        p_a = float(value)
        if not 0 <= p_a <= 1:
            raise ValueError(f"Invalid win probability for {match_id}: {value}")
        by_match[match_id] = p_a

    correct = 0
    losses = []
    for match_id, outcome in outcomes:
        if match_id not in by_match:
            raise ValueError(f"Missing prediction or repeated outcome for {match_id}")
        if outcome not in (0, 1, "0", "1"):
            raise ValueError(f"Invalid A-win label for {match_id}: {outcome}")
        p_a = by_match.pop(match_id)
        a_won = int(outcome)
        correct += (p_a >= 0.5) == a_won
        if a_won:
            losses.append(-log(p_a) if p_a > 0 else inf)
        else:
            losses.append(-log1p(-p_a) if p_a < 1 else inf)

    if by_match:
        raise ValueError(f"{len(by_match)} predictions have no matching outcome")
    count = len(losses)
    if not count:
        raise ValueError("At least one match is required")
    return {
        "matches": count,
        "correct": correct,
        "accuracy": correct / count,
        "mean_log_loss": fsum(losses) / count,
    }
