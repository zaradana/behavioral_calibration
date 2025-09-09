from typing import Any, Dict, List, Tuple

from schema import ItemEval
from utils import correctness_heuristic


def payoff_value(correct: bool, decision: str, t: float) -> float:
    if decision == "idk":
        return 0.0
    if correct:
        return 1.0
    return -t / (1.0 - t)


async def evaluate_model(
    data: List[Dict[str, Any]], targets: List[float], call_model_func
) -> Dict[float, List[ItemEval]]:
    """Evaluate a model across different targets. call_model_func should be async."""
    out: Dict[float, List[ItemEval]] = {}
    for t in targets:
        rows: List[ItemEval] = []
        for item in data:
            obj = await call_model_func(item["prompt"], t)
            decision = obj["decision"]
            ans = obj["answer"]
            conf = float(obj["confidence"])
            is_correct = (
                correctness_heuristic(ans, item["expect_substrings"])
                if decision == "answer"
                else False
            )
            pay = payoff_value(is_correct, decision, t)
            rows.append(
                ItemEval(
                    item_id=item["id"],
                    decision=decision,
                    answer=ans,
                    confidence=conf,
                    correct=is_correct,
                    payoff_behavioral=pay,
                )
            )
        out[t] = rows
    return out


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
    avg_pay = sum(r.payoff_behavioral for r in rows) / n if n else 0.0
    return acc, cov, avg_pay


def inconsistency_rates(rows: List[ItemEval], t: float) -> Dict[str, float]:
    """
    Measures disagreement between channels at target t:
      - idk_high_conf: fraction with decision=='idk' but confidence >= t
      - ans_low_conf:  fraction with decision=='answer' but confidence < t
    """
    n = len(rows) or 1
    idk_high = sum(1 for r in rows if r.decision == "idk" and r.confidence >= t) / n
    ans_low = sum(1 for r in rows if r.decision == "answer" and r.confidence < t) / n
    return {"idk_high_conf": idk_high, "answer_low_conf": ans_low}
