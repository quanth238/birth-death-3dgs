import unittest

import torch

from utils.adc_utils import (
    rank_normalize_masked,
    restrict_to_topk_shortlist,
    select_low_score_death_mask,
    select_score_threshold_mask,
    select_topk_mask,
)


class SelectTopKMaskTest(unittest.TestCase):
    def test_score_threshold_selects_all_finite_values_above_threshold(self):
        scores = torch.tensor([-1.0, 0.0, 0.2, float("nan"), float("inf")])
        selected = select_score_threshold_mask(scores, threshold=0.0)
        self.assertEqual(selected.tolist(), [False, False, True, False, False])

    def test_score_threshold_is_strict(self):
        selected = select_score_threshold_mask(torch.tensor([0.1, 0.2]), threshold=0.1)
        self.assertEqual(selected.tolist(), [False, True])

    def test_selects_highest_eligible_scores(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 1.2, 0.7])
        eligible = torch.tensor([True, True, True, False, True])
        selected = select_topk_mask(scores, eligible, quota=2)
        self.assertEqual(selected.tolist(), [False, True, True, False, False])

    def test_quota_larger_than_pool_returns_all_eligible(self):
        scores = torch.tensor([0.1, 0.2, 0.3])
        eligible = torch.tensor([True, False, True])
        selected = select_topk_mask(scores, eligible, quota=5)
        self.assertEqual(selected.tolist(), eligible.tolist())

    def test_zero_quota_returns_empty_mask(self):
        scores = torch.tensor([0.1, 0.2])
        eligible = torch.tensor([True, True])
        selected = select_topk_mask(scores, eligible, quota=0)
        self.assertFalse(selected.any().item())

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            select_topk_mask(torch.tensor([0.1]), torch.tensor([True, False]), quota=1)

    def test_shortlist_keeps_top_multiple_of_quota(self):
        scores = torch.tensor([0.1, 0.9, 0.8, 1.2, 0.7, 0.6])
        eligible = torch.tensor([True, True, True, False, True, True])
        shortlisted = restrict_to_topk_shortlist(scores, eligible, quota=2, shortlist_factor=2)
        self.assertEqual(shortlisted.tolist(), [False, True, True, False, True, True])

    def test_shortlist_disabled_returns_all_eligible(self):
        scores = torch.tensor([0.1, 0.9, 0.8])
        eligible = torch.tensor([True, False, True])
        shortlisted = restrict_to_topk_shortlist(scores, eligible, quota=1, shortlist_factor=0)
        self.assertEqual(shortlisted.tolist(), eligible.tolist())

    def test_zero_quota_shortlist_is_empty(self):
        scores = torch.tensor([0.1, 0.9])
        eligible = torch.tensor([True, True])
        shortlisted = restrict_to_topk_shortlist(scores, eligible, quota=0, shortlist_factor=2)
        self.assertFalse(shortlisted.any().item())

    def test_rank_normalization_is_masked_and_ordered(self):
        scores = torch.tensor([3.0, 10.0, 1.0, 2.0])
        eligible = torch.tensor([True, False, True, True])
        normalized = rank_normalize_masked(scores, eligible)
        self.assertTrue(torch.allclose(normalized, torch.tensor([1.0, 0.0, 0.0, 0.5])))

    def test_rank_normalization_single_entry_is_one(self):
        scores = torch.tensor([3.0, 1.0])
        eligible = torch.tensor([False, True])
        normalized = rank_normalize_masked(scores, eligible)
        self.assertEqual(normalized.tolist(), [0.0, 1.0])

    def test_low_score_death_preserves_vanilla_prune_count(self):
        scores = torch.tensor([4.0, -3.0, -2.0, 1.0, -1.0])
        opacity = torch.tensor([0.002, 0.010, 0.004, 0.020, 0.012])
        selected = select_low_score_death_mask(scores, opacity, 0.005, 0.015)
        self.assertEqual(int(selected.sum().item()), 2)
        self.assertEqual(selected.tolist(), [False, True, True, False, False])

    def test_low_score_death_is_empty_when_vanilla_would_not_prune(self):
        selected = select_low_score_death_mask(
            torch.tensor([-2.0, -1.0]),
            torch.tensor([0.01, 0.02]),
            0.005,
            0.015,
        )
        self.assertFalse(selected.any().item())

    def test_low_score_death_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            select_low_score_death_mask(
                torch.tensor([1.0]),
                torch.tensor([0.001, 0.002]),
                0.005,
                0.015,
            )


if __name__ == "__main__":
    unittest.main()
