"""
Evaluation metrics: Exact Match (EM), F1, Accuracy — matching the paper's metrics.
"""
import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def accuracy(prediction, ground_truth):
    """Relaxed accuracy: ground truth substring in prediction or vice versa."""
    p, g = normalize_answer(prediction), normalize_answer(ground_truth)
    return int(g in p or p in g)


def evaluate_results(results):
    """Compute aggregate metrics from a list of result dicts."""
    n = len(results)
    if n == 0:
        return {}
    return {
        "EM": sum(r["em"] for r in results) / n * 100,
        "F1": sum(r["f1"] for r in results) / n * 100,
        "Acc": sum(r["acc"] for r in results) / n * 100,
        "Avg_Steps": sum(r["steps"] for r in results) / n,
        "Avg_Time": sum(r["time"] for r in results) / n,
        "N": n,
    }
