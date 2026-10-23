import random
import tempfile
import unittest

import torch

from utils.first_variation import (
    aggregate_vanilla_proposal_utilities,
    select_positive_weighted_mask,
    select_positive_weighted_with_ranked_fallback,
    select_threshold_with_growth_limit,
    select_score_views,
    stabilize_first_variation_score,
    taming_population_target,
    load_birth_quota_schedule,
)


class FirstVariationUtilityTest(unittest.TestCase):
    def test_vanilla_proposal_utility_accounts_for_parent_removal(self):
        base = torch.tensor([2.0, 3.0, 5.0])
        children = torch.tensor([7.0, 11.0, 13.0])
        parents = torch.tensor([0, 1, 1])
        candidates = torch.tensor([True, True, False])
        splits = torch.tensor([False, True, False])

        utility = aggregate_vanilla_proposal_utilities(
            base, children, parents, candidates, splits
        )

        self.assertTrue(torch.equal(utility, torch.tensor([7.0, 21.0, 0.0])))

    def test_vanilla_proposal_utility_rejects_mismatched_children(self):
        with self.assertRaises(ValueError):
            aggregate_vanilla_proposal_utilities(
                torch.ones(2),
                torch.ones(2),
                torch.tensor([0]),
                torch.tensor([True, True]),
                torch.tensor([False, False]),
            )

    def test_score_view_selection_is_deterministic_and_rng_isolated(self):
        cameras = list(range(20))
        random.seed(123)
        state_before = random.getstate()
        first = select_score_views(cameras, num_views=5, seed=9)
        state_after = random.getstate()
        second = select_score_views(cameras, num_views=5, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(state_before, state_after)

    def test_ema_and_absolute_quantile_clipping(self):
        raw = torch.tensor([1.0, 3.0])
        previous = torch.tensor([0.0, 2.0])
        ema, stabilized, metadata = stabilize_first_variation_score(
            raw,
            previous,
            ema_decay=0.5,
            clip_quantile=0.5,
        )
        self.assertTrue(torch.allclose(ema, torch.tensor([0.5, 2.5])))
        self.assertTrue(torch.allclose(stabilized, torch.tensor([0.5, 1.5])))
        self.assertAlmostEqual(metadata["score_clip_value"], 1.5)
        self.assertFalse(metadata["score_ema_reset"])

    def test_shape_change_resets_ema(self):
        raw = torch.tensor([1.0, -1.0, 2.0])
        previous = torch.tensor([7.0])
        ema, _, metadata = stabilize_first_variation_score(raw, previous, 0.9, 0.99)
        self.assertTrue(torch.equal(ema, raw))
        self.assertTrue(metadata["score_ema_reset"])

    def test_positive_weighted_selection_is_exact_and_rng_isolated(self):
        scores = torch.tensor([4.0, 3.0, 2.0, 1.0, -1.0])
        eligible = torch.tensor([True, True, True, True, True])
        torch.manual_seed(123)
        state_before = torch.get_rng_state()
        first, metadata = select_positive_weighted_mask(scores, eligible, quota=3, seed=17)
        state_after = torch.get_rng_state()
        second, _ = select_positive_weighted_mask(scores, eligible, quota=3, seed=17)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state_before, state_after))
        self.assertEqual(int(first.sum()), 3)
        self.assertTrue(torch.all(scores[first] > 0))
        self.assertFalse(metadata["positive_weight_fallback"])

    def test_positive_weighted_selection_logs_zero_weight_fallback(self):
        scores = torch.tensor([2.0, -1.0, 0.0, -3.0])
        eligible = torch.tensor([True, True, True, False])
        selected, metadata = select_positive_weighted_mask(scores, eligible, quota=3, seed=5)
        self.assertEqual(int(selected.sum()), 3)
        self.assertTrue(torch.all(torch.logical_or(~selected, eligible)))
        self.assertEqual(metadata["positive_weight_count"], 1)
        self.assertTrue(metadata["positive_weight_fallback"])

    def test_threshold_growth_limit_samples_exact_fraction_by_positive_q(self):
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, -1.0, float("nan"), 8.0, 7.0, 6.0])
        eligible = torch.tensor([True, True, True, True, True, True, True, False, False, False])
        torch.manual_seed(123)
        state_before = torch.get_rng_state()
        first, metadata = select_threshold_with_growth_limit(
            scores, eligible, threshold=0.0, max_growth_fraction=0.2, seed=19
        )
        state_after = torch.get_rng_state()
        second, _ = select_threshold_with_growth_limit(
            scores, eligible, threshold=0.0, max_growth_fraction=0.2, seed=19
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state_before, state_after))
        self.assertEqual(int(first.sum()), 2)
        self.assertTrue(torch.all(scores[first] > 0))
        self.assertEqual(metadata["threshold_candidate_count"], 5)
        self.assertEqual(metadata["max_growth_birth_limit"], 2)
        self.assertTrue(metadata["growth_limit_applied"])
        self.assertEqual(
            metadata["birth_selection"],
            "q_proportional_without_replacement_under_growth_limit",
        )

    def test_zero_threshold_growth_fraction_preserves_select_all_behavior(self):
        scores = torch.tensor([2.0, 0.0, -1.0, 3.0])
        selected, metadata = select_threshold_with_growth_limit(
            scores,
            torch.ones(4, dtype=torch.bool),
            threshold=0.0,
            max_growth_fraction=0.0,
            seed=1,
        )
        self.assertTrue(torch.equal(selected, torch.tensor([True, False, False, True])))
        self.assertFalse(metadata["growth_limit_enabled"])
        self.assertFalse(metadata["growth_limit_applied"])

    def test_nonzero_margin_growth_limit_samples_only_confident_candidates(self):
        scores = torch.tensor([1.1e-7, 1.5e-7, 3.0e-7, 8.0e-7, 1.0e-7, 9.0e-8])
        selected, metadata = select_threshold_with_growth_limit(
            scores,
            torch.ones(6, dtype=torch.bool),
            threshold=1.0e-7,
            max_growth_fraction=0.5,
            seed=23,
        )

        self.assertEqual(int(selected.sum()), 3)
        self.assertTrue(torch.all(scores[selected] > 1.0e-7))
        self.assertEqual(metadata["threshold_candidate_count"], 4)
        self.assertEqual(metadata["max_growth_birth_limit"], 3)
        self.assertTrue(metadata["growth_limit_applied"])
        self.assertFalse(metadata["positive_weight_fallback"])

    def test_threshold_growth_limit_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            select_threshold_with_growth_limit(
                torch.ones(2),
                torch.ones(2, dtype=torch.bool),
                threshold=0.0,
                max_growth_fraction=1.01,
                seed=0,
            )

    def test_population_budget_composes_with_growth_limit(self):
        scores = torch.arange(10, 0, -1, dtype=torch.float32)
        selected, metadata = select_threshold_with_growth_limit(
            scores,
            torch.ones(10, dtype=torch.bool),
            threshold=0.0,
            max_growth_fraction=0.5,
            seed=31,
            max_births=2,
        )

        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(metadata["max_growth_birth_limit"], 5)
        self.assertEqual(metadata["population_budget_birth_limit"], 2)
        self.assertEqual(metadata["effective_birth_limit"], 2)
        self.assertTrue(metadata["growth_limit_applied"])
        self.assertTrue(metadata["population_budget_limit_applied"])

    def test_taming_population_target_is_monotonic_and_exact(self):
        targets = [
            taming_population_target(100, 500, iteration, 500, 1500)
            for iteration in (500, 750, 1000, 1250, 1500)
        ]

        self.assertEqual(targets[0], 100)
        self.assertEqual(targets[2], 400)
        self.assertEqual(targets[-1], 500)
        self.assertEqual(targets, sorted(targets))

    def test_taming_population_target_rejects_invalid_budget(self):
        with self.assertRaises(ValueError):
            taming_population_target(100, 99, 600, 500, 1500)

    def test_positive_weighted_ranked_fallback_uses_native_order(self):
        scores = torch.tensor([2.0, -1.0, 0.0, -3.0])
        native_scores = torch.tensor([0.0, 4.0, 3.0, 100.0])
        eligible = torch.tensor([True, True, True, False])
        selected, metadata = select_positive_weighted_with_ranked_fallback(
            scores,
            eligible,
            native_scores,
            quota=2,
            seed=5,
        )
        self.assertTrue(torch.equal(selected, torch.tensor([True, True, False, False])))
        self.assertEqual(metadata["positive_weight_count"], 1)
        self.assertTrue(metadata["positive_weight_fallback"])
        self.assertEqual(metadata["ranked_fallback_count"], 1)
        self.assertEqual(metadata["ranked_fallback"], "vanilla_position_gradient_top_r")

    def test_positive_weighted_ranked_selection_is_rng_isolated(self):
        scores = torch.tensor([4.0, 3.0, 2.0, 1.0, -1.0])
        native_scores = torch.arange(5, dtype=torch.float32)
        eligible = torch.ones(5, dtype=torch.bool)
        torch.manual_seed(123)
        state_before = torch.get_rng_state()
        first, metadata = select_positive_weighted_with_ranked_fallback(
            scores,
            eligible,
            native_scores,
            quota=3,
            seed=17,
        )
        state_after = torch.get_rng_state()
        second, _ = select_positive_weighted_with_ranked_fallback(
            scores,
            eligible,
            native_scores,
            quota=3,
            seed=17,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state_before, state_after))
        self.assertFalse(metadata["positive_weight_fallback"])
        self.assertEqual(metadata["ranked_fallback_count"], 0)

    def test_birth_quota_schedule_loader(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            handle.write('{"iteration": 600, "birth_quota": 12}\n')
            handle.write('{"iteration": 700, "birth_quota": 7}\n')
            handle.flush()
            self.assertEqual(load_birth_quota_schedule(handle.name), {600: 12, 700: 7})


if __name__ == "__main__":
    unittest.main()
