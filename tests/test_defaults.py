from argparse import ArgumentParser
import unittest

from arguments import OptimizationParams


class PublicDefaultsTest(unittest.TestCase):
    def test_enabling_controller_uses_documented_configuration(self):
        parser = ArgumentParser()
        options = OptimizationParams(parser)
        config = options.extract(parser.parse_args(["--first_variation_adc"]))
        self.assertTrue(config.first_variation_adc)
        self.assertEqual(config.first_variation_birth_selection, "q_threshold")
        self.assertEqual(config.first_variation_birth_threshold, 1e-7)
        self.assertEqual(config.first_variation_max_growth_fraction, 0.10)
        self.assertEqual(config.first_variation_score_views, 16)
        self.assertEqual(config.first_variation_score_ema, 0.0)
        self.assertEqual(config.first_variation_score_clip_quantile, 0.0)
        self.assertEqual(config.first_variation_death_selection, "standard_opacity")
        self.assertEqual(config.max_gaussians, -1)

    def test_baseline_does_not_enable_first_variation(self):
        parser = ArgumentParser()
        OptimizationParams(parser)
        self.assertFalse(parser.parse_args([]).first_variation_adc)


if __name__ == "__main__":
    unittest.main()
