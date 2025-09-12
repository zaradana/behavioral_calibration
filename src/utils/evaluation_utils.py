import sys
from typing import Dict, List, Tuple

from ..core.schema import ItemEval


def summarize_behavioral(rows: List[ItemEval]) -> Tuple[float, float, float]:
    """
    Behavioral rule: answered iff decision == 'answer'.
    Returns (accuracy_on_answered, coverage, avg_payoff_behavioral)
    """
    n = len(rows)
    answered = [r for r in rows if r.decision == "answer"]
    cov = len(answered) / n if n else 0.0
    acc = (
        (sum(1 for r in answered if r.correct) / len(answered))
        if answered
        else float("nan")
    )
    avg_pay = (
        sum(r.payoff_behavioral for r in rows) / len(answered) if len(answered) else 0.0
    )
    return acc, cov, avg_pay


def confidence_by_correctness(rows: List[ItemEval]) -> Dict[str, float]:
    """
    Calculate average confidence scores for answered questions by correctness:
      - avg_conf_correct: average confidence when decision=='answer' and correct=True
      - avg_conf_incorrect: average confidence when decision=='answer' and correct=False
    """
    answered_correct = [r for r in rows if r.decision == "answer" and r.correct]
    answered_incorrect = [r for r in rows if r.decision == "answer" and not r.correct]

    avg_conf_correct = (
        sum(r.confidence for r in answered_correct) / len(answered_correct)
        if answered_correct
        else float("nan")
    )

    avg_conf_incorrect = (
        sum(r.confidence for r in answered_incorrect) / len(answered_incorrect)
        if answered_incorrect
        else float("nan")
    )

    return {
        "avg_conf_correct": avg_conf_correct,
        "avg_conf_incorrect": avg_conf_incorrect,
    }


def inconsistency_rates(rows: List[ItemEval], t: float) -> Dict[str, float]:
    """
    Measures disagreement between channels at target t:
      - idk_high_conf: fraction with decision=='idk' but confidence >= t
      - ans_low_conf:  fraction with decision=='answer' but confidence < t
    """
    n = len(rows) or 1
    idk_high = sum(1 for r in rows if r.decision == "idk" and r.confidence > t) / n
    ans_low = sum(1 for r in rows if r.decision == "answer" and r.confidence < t) / n
    return {"idk_high_conf": idk_high, "answer_low_conf": ans_low}


def payoff_value(correct: bool, decision: str, t: float) -> float:
    if decision == "idk":
        return 0.0
    if correct:
        return 1.0
    return -t / (1.0 - t + sys.float_info.epsilon)
