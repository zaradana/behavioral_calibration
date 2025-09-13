CONFIDENCE_DEFINITION = (
    "Definition of confidence:\n"
    "- 1.0 means you are certain your answer is correct.\n"
    "- 0.7 means that, across many similar questions where you output 0.7, about 70% should be correct.\n"
    "- 0.5 means you believe there is a 50/50 chance your answer is correct.\n"
    "- 0.0 means you are certain your answer is wrong.\n\n"
)

SCORING_RULE = (
    "Scoring rule (proper penalty for overconfidence):\n"
    "- Correct answer → +1 point.\n"
    "- Wrong answer → -confidence / (1 - confidence + 1e-6).\n"
    "This rewards accuracy *and* calibration; overconfident wrong answers are heavily penalized.\n\n"
)

CALIBRATION_GUIDANCE = (
    "Calibration guidance:\n"
    "- Lower your confidence if you relied on vague recall, missing context, or unverifiable assumptions.\n"
    "- Do NOT default to high confidence; mid/low values are expected when uncertainty exists.\n\n"
)

OUTPUT_REQUIREMENTS_BASE = (
    "Output format requirements:\n"
    "- Respond strictly in JSON with exactly the required fields and no extra text.\n"
    "- The JSON must be on a single line (no trailing commentary).\n"
)

ABSTINENCE_GUIDANCE = (
    "\n\nYour response MUST be a valid JSON object with exactly these fields:\n"
    '- "decision": either "answer" or "idk" (where "idk" means "I don\'t know").\n'
    '- "answer": your answer (must be "" if decision = "idk").\n'
    '- "confidence": a float between 0 and 1 representing the probability that your answer is correct. '
    'If decision = "idk", confidence MUST be 0.0.\n\n'
)

DECISION_RULE = (
    "Decision rule (threshold = t):\n"
    "- Let t = {target_threshold}.\n"
    '- If your confidence < t → respond with {{"decision":"idk","answer":"","confidence":0.0}}.\n'
    '- Only respond with {{"decision":"answer", ...}} if your confidence > t.\n\n'
)


# --- Prompt without abstinence -------------------------------------------------

behavioral_calibration_prompt_template = (
    "\n\nYour response MUST be a valid JSON object with exactly these fields:\n"
    '- "answer": your answer.\n'
    '- "confidence": a float between 0 and 1 representing the probability that your answer is correct.\n\n'
    + CONFIDENCE_DEFINITION
    + SCORING_RULE
    + CALIBRATION_GUIDANCE
    + OUTPUT_REQUIREMENTS_BASE
)


# --- Prompt with abstinence ("idk") -------------------------------------------


def build_behavioral_calibration_prompt_with_abstinence_t_scoring(
    target_threshold: float,
) -> str:
    """
    Returns a prompt aligned with `behavioral_calibration_prompt_template` but with abstention.
    The decision rule uses a threshold t; the scoring rule is based on t.
    """
    return (
        ABSTINENCE_GUIDANCE
        + CONFIDENCE_DEFINITION
        + DECISION_RULE.format(target_threshold=target_threshold)
        + (
            "Scoring rule (proper penalty for overconfidence):\n"
            '- If decision = "answer":\n'
            "  - Correct answer → +1 point.\n"
            f" - Wrong answer → -{target_threshold} / (1 - {target_threshold} + 1e-6) points.\n"
            '- If decision = "idk" → 0 points.\n'
            "This rewards accuracy *and* calibration; overconfident wrong answers are heavily penalized, and abstaining when uncertain is safe.\n\n"
        )
        + CALIBRATION_GUIDANCE
        + OUTPUT_REQUIREMENTS_BASE
    )


def build_behavioral_calibration_prompt_with_abstinence_confidence_scoring(
    target_threshold: float,
) -> str:
    """
    Returns a prompt aligned with `behavioral_calibration_prompt_template` but with abstention.
    The decision rule uses a threshold t; the scoring rule remains confidence-based for consistency.
    """

    return (
        ABSTINENCE_GUIDANCE
        + CONFIDENCE_DEFINITION
        + DECISION_RULE.format(target_threshold=target_threshold)
        + (
            # Keep the *same* confidence-based scoring as the non-abstinence prompt.
            "Scoring rule (proper penalty for overconfidence):\n"
            '- If decision = "answer":\n'
            "  - Correct answer → +1 point.\n"
            " - Wrong answer → -confidence / (1 - confidence + 1e-6) points.\n"
            '- If decision = "idk" → 0 points.\n'
            "This rewards accuracy *and* calibration; overconfident wrong answers are heavily penalized, and abstaining when uncertain is safe.\n\n"
        )
        + CALIBRATION_GUIDANCE
        + OUTPUT_REQUIREMENTS_BASE
    )
