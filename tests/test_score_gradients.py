"""Check the actual score wrapper using a small CPU reference compositor.

These tests exercise gradient wiring, not the CUDA rasterizer. GPU validation
against that rasterizer is provided by scripts/validate_first_variation_score.py.
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.first_variation import (
    compute_first_variation_score,
    first_variation_image_objective,
)


class ReferenceRasterizer:
    def __init__(self, raster_settings):
        self.settings = raster_settings

    def __call__(self, **inputs):
        alpha = inputs["opacities"].flatten()
        colors = inputs.get("colors_precomp")
        if colors is None:
            colors = inputs["shs"][:, 0, :]
        # Depend on all continuous inputs so an accidental missing detach is visible.
        geometry = inputs["means3D"].sum(dim=1)
        geometry = geometry + inputs["scales"].sum(dim=1)
        geometry = geometry + inputs["rotations"].sum(dim=1)
        colors = colors + 0.01 * geometry[:, None]
        prefix = torch.cat((torch.ones_like(alpha[:1]), 1 - alpha[:-1]))
        transmittance = torch.cumprod(prefix, dim=0)
        rgb = (transmittance[:, None] * alpha[:, None] * colors).sum(dim=0)
        rgb = rgb + (1 - alpha).prod() * self.settings.bg
        image = rgb[:, None, None].expand(3, 2, 2)
        return image, torch.ones_like(alpha), torch.zeros_like(image[:1])


class TinyScene(torch.nn.Module):
    active_sh_degree = 0
    max_sh_degree = 0

    def __init__(self):
        super().__init__()
        self.xyz = torch.nn.Parameter(torch.full((2, 3), 0.1, dtype=torch.float64))
        self.log_scale = torch.nn.Parameter(torch.full((2, 3), -2.0, dtype=torch.float64))
        self.rotation = torch.nn.Parameter(torch.tensor(
            [[1.0, 0.1, 0.0, 0.0], [1.0, 0.0, 0.1, 0.0]], dtype=torch.float64
        ))
        self.logit = torch.nn.Parameter(torch.tensor([[-0.4], [0.7]], dtype=torch.float64))
        self.features = torch.nn.Parameter(torch.tensor(
            [[[0.7, 0.2, 0.1]], [[0.1, 0.4, 0.7]]], dtype=torch.float64
        ))

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scaling(self):
        return self.log_scale.exp()

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotation, dim=1)

    @property
    def get_opacity(self):
        return self.logit.sigmoid()

    @property
    def get_features(self):
        return self.features


class MassGradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = ModuleType("diff_gaussian_rasterization")
        backend.GaussianRasterizationSettings = SimpleNamespace
        backend.GaussianRasterizer = ReferenceRasterizer
        source = Path(__file__).resolve().parents[1] / "gaussian_renderer" / "__init__.py"
        spec = importlib.util.spec_from_file_location("reference_score_renderer", source)
        cls.renderer = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"diff_gaussian_rasterization": backend}):
            spec.loader.exec_module(cls.renderer)

    def setUp(self):
        self.scene = TinyScene()
        self.pipe = SimpleNamespace(
            compute_cov3D_python=False, convert_SHs_python=False,
            debug=False, antialiasing=False,
        )
        self.background = torch.tensor([0.1, 0.05, 0.2], dtype=torch.float64)
        self.views = [SimpleNamespace(
            FoVx=1.0, FoVy=1.0, image_height=2, image_width=2,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3), alpha_mask=None,
            original_image=torch.full((3, 2, 2), value, dtype=torch.float64),
            image_name=f"view_{i}",
        ) for i, value in enumerate([0.15, 0.65])]

    def score(self, views=None):
        with patch.dict("sys.modules", {"gaussian_renderer": self.renderer}):
            return compute_first_variation_score(
                self.views if views is None else views,
                self.scene, self.pipe, self.background, 0.2,
            )[0]

    def objective(self, mass):
        values = []
        for view in self.views:
            image = self.renderer.render_mass_gated(
                view, self.scene, self.pipe, self.background, mass
            )["render"]
            values.append(first_variation_image_objective(image, view.original_image, 0.2))
        return torch.stack(values).mean()

    def test_unit_mass_matches_normal_forward_and_detaches_scene(self):
        view = self.views[0]
        normal = self.renderer.render(view, self.scene, self.pipe, self.background)["render"]
        mass = torch.ones_like(self.scene.get_opacity, requires_grad=True)
        gated = self.renderer.render_mass_gated(
            view, self.scene, self.pipe, self.background, mass
        )["render"]
        torch.testing.assert_close(gated, normal)
        gated.sum().backward()
        self.assertIsNotNone(mass.grad)
        self.assertTrue(all(p.grad is None for p in self.scene.parameters()))

    def test_score_matches_opacity_chain_rule_and_finite_difference(self):
        score = self.score()
        alpha = self.scene.get_opacity.detach().requires_grad_(True)
        losses = []
        rasterizer = ReferenceRasterizer(SimpleNamespace(bg=self.background))
        for view in self.views:
            image, _, _ = rasterizer(
                opacities=alpha, means3D=self.scene.get_xyz.detach(),
                scales=self.scene.get_scaling.detach(),
                rotations=self.scene.get_rotation.detach(),
                shs=self.scene.get_features.detach(),
            )
            losses.append(first_variation_image_objective(image, view.original_image, 0.2))
        opacity_grad, = torch.autograd.grad(torch.stack(losses).mean(), alpha)
        torch.testing.assert_close(score, (-alpha * opacity_grad).flatten())
        epsilon = 1e-5
        for i in range(score.numel()):
            plus = torch.ones_like(alpha)
            minus = torch.ones_like(alpha)
            plus[i] += epsilon
            minus[i] -= epsilon
            numerical = -(self.objective(plus) - self.objective(minus)) / (2 * epsilon)
            torch.testing.assert_close(score[i], numerical, atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(score, (self.score(self.views[:1]) + self.score(self.views[1:])) / 2)

    def test_score_does_not_change_existing_training_gradients(self):
        for parameter in self.scene.parameters():
            parameter.grad = torch.full_like(parameter, 3.0)
        self.score()
        for parameter in self.scene.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 3.0))

    def test_empty_score_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one score view"):
            self.score([])


if __name__ == "__main__":
    unittest.main()
