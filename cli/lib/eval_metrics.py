"""
eval_metrics.py
---------------
Information Retrieval evaluation metrics for the Hoopla movie recommendation system.

All metrics use *binary* relevance judgements (a document is either relevant or not),
matching the format of data/golden_dataset.json.

Terminology
-----------
retrieved   : ordered list of document titles returned by the system (position 0 = rank 1)
relevant    : set (or list) of titles that are ground-truth relevant for the query
k           : cut-off depth — we only look at the top-k retrieved results
"""

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rel_vector(retrieved: Sequence[str], relevant: set[str], k: int) -> list[int]:
    """Return a binary relevance vector for the top-k retrieved docs."""
    return [1 if r in relevant else 0 for r in retrieved[:k]]


# ---------------------------------------------------------------------------
# Precision @ k
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved documents that are relevant.

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth.

    Returns:
        Precision@k in [0, 1].
    """
    if k <= 0:
        return 0.0

    hits = sum(_rel_vector(retrieved, relevant, k))
    return hits / min(k, len(retrieved)) if retrieved else 0.0


# ---------------------------------------------------------------------------
# Recall @ k
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant documents that appear in the top-k results.

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth.

    Returns:
        Recall@k in [0, 1].
    """
    if not relevant:
        return 0.0

    hits = sum(_rel_vector(retrieved, relevant, k))
    return hits / len(relevant)


# ---------------------------------------------------------------------------
# F1 @ k
# ---------------------------------------------------------------------------

def f1_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of Precision@k and Recall@k.

    Returns 0.0 when both precision and recall are 0 (avoids division by zero).

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth.

    Returns:
        F1@k in [0, 1].
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# NDCG @ k  (Normalised Discounted Cumulative Gain)
# ---------------------------------------------------------------------------

def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Uses binary relevance gains (0 or 1).  Higher positions are worth more
    because we discount by log2(rank + 1).

    Formula::

        DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)
        IDCG@k = DCG of ideal ranking (all relevant docs first)
        NDCG@k = DCG@k / IDCG@k

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth.

    Returns:
        NDCG@k in [0, 1].  Returns 0.0 if there are no relevant documents.
    """
    if not relevant or k <= 0:
        return 0.0

    rel_vector = _rel_vector(retrieved, relevant, k)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rel_vector))

    # Ideal DCG: place all relevant docs at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# Reciprocal Rank  (building block for MRR)
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result.

    Searches through *all* retrieved documents (no cut-off), so MRR is not
    truncated.  If no relevant document is found, returns 0.0.

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.

    Returns:
        1 / rank_first_hit, or 0.0 if no hit is found.
    """
    for rank, title in enumerate(retrieved, start=1):
        if title in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Average Precision @ k  (building block for MAP)
# ---------------------------------------------------------------------------

def average_precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Average Precision at cut-off k.

    Computes the average of Precision@i for each position i (1 ≤ i ≤ k)
    where a relevant document was retrieved, then normalises by the number
    of relevant documents (up to k).

    Formula::

        AP@k = Σ_{i=1}^{k} (P@i × rel_i) / min(|relevant|, k)

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth.

    Returns:
        AP@k in [0, 1].  Returns 0.0 if there are no relevant documents.
    """
    if not relevant or k <= 0:
        return 0.0

    hits = 0
    cumulative_precision = 0.0

    for i, title in enumerate(retrieved[:k], start=1):
        if title in relevant:
            hits += 1
            cumulative_precision += hits / i  # P@i at each relevant hit

    normaliser = min(len(relevant), k)
    return cumulative_precision / normaliser if normaliser > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> dict[str, float]:
    """Compute every supported IR metric for a single query.

    Args:
        retrieved: Ordered list of retrieved document titles (rank 1 first).
        relevant:  Set of ground-truth relevant document titles.
        k:         Cut-off depth applied to Precision, Recall, F1, NDCG, AP.

    Returns:
        Dictionary mapping metric name to its value::

            {
                "precision": float,
                "recall":    float,
                "f1":        float,
                "ndcg":      float,
                "rr":        float,   # reciprocal rank (no cut-off)
                "ap":        float,   # average precision @ k
            }
    """
    return {
        "precision": precision_at_k(retrieved, relevant, k),
        "recall":    recall_at_k(retrieved, relevant, k),
        "f1":        f1_at_k(retrieved, relevant, k),
        "ndcg":      ndcg_at_k(retrieved, relevant, k),
        "rr":        reciprocal_rank(retrieved, relevant),
        "ap":        average_precision_at_k(retrieved, relevant, k),
    }
