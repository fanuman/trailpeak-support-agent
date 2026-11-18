"""Compare an evaluation run with the frozen baseline and report real drops.

The baseline (eval_results/baselines/v1.json) holds the mean of 4 runs of the LLM-judged eval on
golden set v1. LLM judges vary from run to run: on unchanged code the averages moved by about 0.01
(standard deviation 0.007-0.011, largest gap between runs 0.024). REGRESSION_THRESHOLD sits at about
3-4 times that noise: a smaller drop could be chance, a larger one is very likely real.
This file never writes a baseline: a new one is frozen on purpose, after measuring the noise again.
"""
import json

BASELINE_PATH = "eval_results/baselines/v1.json"
REGRESSION_THRESHOLD = 0.04


def load_baseline(path=BASELINE_PATH):
    with open(path) as f:
        return json.load(f)


def check_regression(current_scores: dict, baseline: dict | None = None,
                     threshold: float = REGRESSION_THRESHOLD) -> list[str]:
    """One message per metric that dropped more than `threshold` below the baseline mean (none = no regression)."""
    reference = (baseline or load_baseline())["llm"]["mean"]
    regressions = []
    for metric, base in reference.items():
        current = current_scores[metric]          # a missing metric fails loudly instead of passing
        if current < base - threshold:
            regressions.append(f"{metric}: baseline {base:.3f} -> now {current:.3f} "
                               f"(dropped {base - current:.3f}, more than {threshold})")
    return regressions
