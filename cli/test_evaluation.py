"""
test_evaluation.py
------------------
Unit tests for cli/lib/eval_metrics.py.

Run with:
    cd /Users/sumaymittal/Desktop/hoopla
    uv run pytest cli/test_evaluation.py -v
"""

import math
import sys
import os

# Make sure we can import from cli/lib without the full package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pytest
from lib.eval_metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    reciprocal_rank,
    average_precision_at_k,
    compute_all_metrics,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RELEVANT = {"A", "B", "C"}

# Perfect retrieval — all 3 relevant docs in top-3
PERFECT_3   = ["A", "B", "C"]

# Perfect retrieval of 5 (3 relevant + 2 extra)
PERFECT_5   = ["A", "B", "C", "D", "E"]

# No hits at all
ZERO        = ["X", "Y", "Z", "W", "V"]

# One hit at position 1
FIRST_HIT   = ["A", "X", "Y", "Z", "W"]

# One hit at position 3
THIRD_HIT   = ["X", "Y", "A", "Z", "W"]

# Two hits at positions 2 and 4
TWO_HITS    = ["X", "A", "Y", "B", "Z"]

# All 3 relevant docs present but in reverse order (worst ranking)
REVERSED    = ["C", "B", "A"]


# ===========================================================================
# precision_at_k
# ===========================================================================

class TestPrecisionAtK:
    def test_perfect_all_relevant(self):
        assert precision_at_k(PERFECT_3, RELEVANT, k=3) == pytest.approx(1.0)

    def test_perfect_with_extra(self):
        # top-3 from ["A","B","C","D","E"] → 3 hits / 3 = 1.0
        assert precision_at_k(PERFECT_5, RELEVANT, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert precision_at_k(ZERO, RELEVANT, k=5) == pytest.approx(0.0)

    def test_one_hit_at_start(self):
        # 1 hit / 5 retrieved = 0.2
        assert precision_at_k(FIRST_HIT, RELEVANT, k=5) == pytest.approx(0.2)

    def test_two_hits(self):
        # 2 hits / 5 retrieved = 0.4
        assert precision_at_k(TWO_HITS, RELEVANT, k=5) == pytest.approx(0.4)

    def test_k_larger_than_retrieved(self):
        # k=10, but only 3 docs returned → denominator = 3
        assert precision_at_k(PERFECT_3, RELEVANT, k=10) == pytest.approx(1.0)

    def test_empty_retrieved(self):
        assert precision_at_k([], RELEVANT, k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        assert precision_at_k(PERFECT_3, RELEVANT, k=0) == pytest.approx(0.0)


# ===========================================================================
# recall_at_k
# ===========================================================================

class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(PERFECT_3, RELEVANT, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert recall_at_k(ZERO, RELEVANT, k=5) == pytest.approx(0.0)

    def test_one_of_three(self):
        # 1 hit / 3 relevant = 0.333...
        assert recall_at_k(FIRST_HIT, RELEVANT, k=5) == pytest.approx(1 / 3, rel=1e-4)

    def test_two_of_three(self):
        assert recall_at_k(TWO_HITS, RELEVANT, k=5) == pytest.approx(2 / 3, rel=1e-4)

    def test_empty_relevant(self):
        assert recall_at_k(PERFECT_3, set(), k=5) == pytest.approx(0.0)

    def test_k_limits_recall(self):
        # Only look at k=1 from ["A","B","C"] → 1 hit / 3 relevant = 0.333
        assert recall_at_k(PERFECT_3, RELEVANT, k=1) == pytest.approx(1 / 3, rel=1e-4)


# ===========================================================================
# f1_at_k
# ===========================================================================

class TestF1AtK:
    def test_perfect(self):
        assert f1_at_k(PERFECT_3, RELEVANT, k=3) == pytest.approx(1.0)

    def test_zero_precision_zero_recall(self):
        assert f1_at_k(ZERO, RELEVANT, k=5) == pytest.approx(0.0)

    def test_harmonic_mean(self):
        p = precision_at_k(FIRST_HIT, RELEVANT, k=5)  # 0.2
        r = recall_at_k(FIRST_HIT, RELEVANT, k=5)      # 0.333
        expected_f1 = 2 * p * r / (p + r)
        assert f1_at_k(FIRST_HIT, RELEVANT, k=5) == pytest.approx(expected_f1, rel=1e-4)


# ===========================================================================
# ndcg_at_k
# ===========================================================================

class TestNdcgAtK:
    def test_perfect_ranking(self):
        # All 3 relevant at positions 1,2,3 → NDCG = 1.0
        assert ndcg_at_k(PERFECT_3, RELEVANT, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert ndcg_at_k(ZERO, RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert ndcg_at_k(PERFECT_3, set(), k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        assert ndcg_at_k(PERFECT_3, RELEVANT, k=0) == pytest.approx(0.0)

    def test_single_hit_at_rank1(self):
        # retrieved = ["A"], relevant = {"A"} → DCG = 1/log2(2) = 1.0, IDCG = 1.0
        assert ndcg_at_k(["A"], {"A"}, k=1) == pytest.approx(1.0)

    def test_single_hit_at_rank2(self):
        # retrieved = ["X", "A"], relevant = {"A"}
        # DCG  = 0/log2(2) + 1/log2(3) = 1/log2(3)
        # IDCG = 1/log2(2) = 1.0   (ideal: A at rank 1)
        dcg  = 1.0 / math.log2(3)
        idcg = 1.0 / math.log2(2)
        assert ndcg_at_k(["X", "A"], {"A"}, k=2) == pytest.approx(dcg / idcg, rel=1e-4)

    def test_higher_ranking_yields_higher_ndcg(self):
        # "A" at rank 1 should give higher NDCG than "A" at rank 3
        ndcg_early = ndcg_at_k(FIRST_HIT, RELEVANT, k=5)
        ndcg_late  = ndcg_at_k(THIRD_HIT, RELEVANT, k=5)
        assert ndcg_early > ndcg_late

    def test_ndcg_between_zero_and_one(self):
        for retrieved in [PERFECT_3, ZERO, FIRST_HIT, THIRD_HIT, TWO_HITS]:
            val = ndcg_at_k(retrieved, RELEVANT, k=5)
            assert 0.0 <= val <= 1.0, f"NDCG out of range for {retrieved}: {val}"


# ===========================================================================
# reciprocal_rank
# ===========================================================================

class TestReciprocalRank:
    def test_first_hit_at_rank1(self):
        assert reciprocal_rank(["A", "X", "Y"], RELEVANT) == pytest.approx(1.0)

    def test_first_hit_at_rank2(self):
        assert reciprocal_rank(["X", "A", "Y"], RELEVANT) == pytest.approx(0.5)

    def test_first_hit_at_rank3(self):
        assert reciprocal_rank(THIRD_HIT, RELEVANT) == pytest.approx(1 / 3, rel=1e-4)

    def test_no_hit(self):
        assert reciprocal_rank(ZERO, RELEVANT) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert reciprocal_rank([], RELEVANT) == pytest.approx(0.0)

    def test_rr_in_range(self):
        for retrieved in [FIRST_HIT, THIRD_HIT, TWO_HITS, ZERO]:
            val = reciprocal_rank(retrieved, RELEVANT)
            assert 0.0 <= val <= 1.0


# ===========================================================================
# average_precision_at_k
# ===========================================================================

class TestAveragePrecisionAtK:
    def test_perfect_ap(self):
        # All 3 relevant at positions 1,2,3
        # AP = (P@1 * 1 + P@2 * 1 + P@3 * 1) / min(3, 3)
        #    = (1/1 + 2/2 + 3/3) / 3 = (1 + 1 + 1) / 3 = 1.0
        assert average_precision_at_k(PERFECT_3, RELEVANT, k=3) == pytest.approx(1.0)

    def test_no_hits(self):
        assert average_precision_at_k(ZERO, RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert average_precision_at_k(PERFECT_3, set(), k=5) == pytest.approx(0.0)

    def test_k_zero(self):
        assert average_precision_at_k(PERFECT_3, RELEVANT, k=0) == pytest.approx(0.0)

    def test_one_hit_at_rank1(self):
        # retrieved = ["A","X","Y","Z","W"], relevant = {"A","B","C"}
        # P@1 = 1, only 1 hit in top-5
        # AP@5 = (1/1) / min(3,5) = 1/3
        assert average_precision_at_k(FIRST_HIT, RELEVANT, k=5) == pytest.approx(1 / 3, rel=1e-4)

    def test_two_hits_precision_at_each_hit(self):
        # retrieved = ["X","A","Y","B","Z"], relevant = {"A","B","C"}
        # Hit at position 2 (A): P@2 = 1/2
        # Hit at position 4 (B): P@4 = 2/4 = 0.5
        # AP@5 = (0.5 + 0.5) / min(3,5) = 1.0 / 3
        ap = average_precision_at_k(TWO_HITS, RELEVANT, k=5)
        assert ap == pytest.approx((0.5 + 0.5) / 3, rel=1e-4)

    def test_ap_in_range(self):
        for retrieved in [PERFECT_3, ZERO, FIRST_HIT, THIRD_HIT, TWO_HITS]:
            val = average_precision_at_k(retrieved, RELEVANT, k=5)
            assert 0.0 <= val <= 1.0, f"AP out of range for {retrieved}: {val}"

    def test_early_hit_better_than_late_hit(self):
        ap_early = average_precision_at_k(FIRST_HIT, RELEVANT, k=5)   # hit at pos 1
        ap_late  = average_precision_at_k(THIRD_HIT, RELEVANT, k=5)   # hit at pos 3
        assert ap_early > ap_late


# ===========================================================================
# compute_all_metrics (integration)
# ===========================================================================

class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        result = compute_all_metrics(PERFECT_3, RELEVANT, k=3)
        expected_keys = {"precision", "recall", "f1", "ndcg", "rr", "ap"}
        assert set(result.keys()) == expected_keys

    def test_perfect_retrieval_all_ones(self):
        result = compute_all_metrics(PERFECT_3, RELEVANT, k=3)
        for key, val in result.items():
            assert val == pytest.approx(1.0), f"{key} should be 1.0 for perfect retrieval"

    def test_zero_retrieval_all_zeros(self):
        result = compute_all_metrics(ZERO, RELEVANT, k=5)
        for key, val in result.items():
            assert val == pytest.approx(0.0), f"{key} should be 0.0 when nothing is retrieved"

    def test_all_values_in_range(self):
        for retrieved in [PERFECT_3, PERFECT_5, ZERO, FIRST_HIT, THIRD_HIT, TWO_HITS]:
            result = compute_all_metrics(retrieved, RELEVANT, k=5)
            for key, val in result.items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] for {retrieved}"
