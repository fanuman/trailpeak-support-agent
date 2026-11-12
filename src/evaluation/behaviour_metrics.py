"""Metrics for yes/no decisions, such as "should the bot refuse this question?".

The positive class is the thing we are trying to catch (here: a question that should be refused).
    TP  caught:          decided yes, and it really was yes
    FP  false alarm:     decided yes, but it was no   (a real customer refused)
    FN  missed:          decided no, but it was yes   (an off-topic question answered)
    TN  correctly left:  decided no, and it was no
A ratio whose bottom is zero (e.g. precision when nothing was refused) is reported as 0.0.
"""


def confusion_counts(decisions, truths):
    """Count TP, FP, FN and TN from two lists of True/False of the same length."""
    if len(decisions) != len(truths):
        raise ValueError("decisions and truths must have the same length")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for decided, truth in zip(decisions, truths):
        if decided and truth:
            counts["tp"] += 1
        elif decided:
            counts["fp"] += 1
        elif truth:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def _ratio(part, whole):
    return part / whole if whole else 0.0


def classification_scores(counts):
    """Precision, recall, F1, accuracy and false positive rate from confusion counts."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
        "false_positive_rate": _ratio(fp, fp + tn),
    }
