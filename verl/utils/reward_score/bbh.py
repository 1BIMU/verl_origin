import re


def extract_answer(response: str) -> str:
    patterns = [
        r"[Tt]he answer is[:\s]*([^\n\.]+)",
        r"[Aa]nswer[:\s]*([^\n\.]+)$",
        r"\*\*([^\*]+)\*\*\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            answer = answer.strip(".,;:\"'")
            return answer
    lines = response.strip().split("\n")
    if lines:
        return lines[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = " ".join(answer.split())
    answer = re.sub(r"[^\w\s]", "", answer)
    return answer.strip()


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    extracted = extract_answer(solution_str)
    pred_normalized = normalize_answer(extracted)
    gt_normalized = normalize_answer(ground_truth)

    if pred_normalized == gt_normalized:
        return 1.0

    gt_letter = re.search(r"^\(?([a-zA-Z])\)?$", ground_truth.strip())
    if gt_letter:
        letter = gt_letter.group(1).lower()
        pred_letter = re.search(r"^\(?([a-zA-Z])\)?$", extracted.strip())
        if pred_letter and pred_letter.group(1).lower() == letter:
            return 1.0

    return 0.0
