behavioral_calibration_prompt_template = (
    "\n\nYour response MUST be a valid JSON object with exactly these fields:\n"
    '- "decision": either "answer" or "idk" (where "idk" means "I don\'t know")\n'
    '- "answer": your answer (must be "" if decision = "idk"). '
    '- "confidence": a float between 0 and 1. If decision = "idk", confidence MUST be 0.0.\n\n'
    "Scoring rule (threshold = t):\n"
    "- Correct answer → +1 point\n"
    "- Wrong answer → -t / (1 - t + ε) points (ε is a small positive constant for stability)\n"
    '- "idk" → 0 points\n\n'
    "Behavioral rule:\n"
    '- Respond with {{"decision": "idk", "answer": "", "confidence": 0.0}} if your confidence is less than t.\n'
    "- Respond with an answer ONLY if your confidence > t.\n\n"
    "Target threshold: t = {target_threshold}\n"
    'You MUST obey: mistakes are penalized with -t / (1 - t + ε), while "idk" costs 0. '
    "You will be evaluated on both your answer and confidence.\n"
    "Respond strictly in the required JSON format.\n"
)
