# Third-party notices

This release builds on the following projects. Original copyright and license
notices are retained in the source and submodules.

- **3D Gaussian Splatting**, Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
  and George Drettakis, Inria / MPII. The training code, scene representation,
  renderer wrapper, and utilities derive from
  [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).
  The research/noncommercial license is included in [LICENSE.md](LICENSE.md).
- **Differentiable Gaussian rasterization**, Inria / MPII. Included as a pinned
  [submodule](https://github.com/graphdeco-inria/diff-gaussian-rasterization),
  with a small `<cstdint>` compatibility patch. Its own license applies.
- **simple-knn**, Bernhard Kerbl / Inria. Included as a pinned
  [submodule](https://gitlab.inria.fr/bkerbl/simple-knn); its own license applies.
- **fused-ssim**, Rahul Goel. Included as a pinned
  [submodule](https://github.com/rahul-goel/fused-ssim); its own license applies.
- **pytorch-ssim**, Evan Su, 2017. `utils/loss_utils.py` includes code from this
  MIT-licensed project, as recorded in the upstream license. A copy is in
  [docs/licenses/pytorch-ssim.txt](docs/licenses/pytorch-ssim.txt).
- **lpips-pytorch**. The `lpipsPyTorch/` implementation is inherited from the
  upstream 3DGS repository and derives from
  [S-aiueo32/lpips-pytorch](https://github.com/S-aiueo32/lpips-pytorch).
  A copy of its BSD 2-Clause license is in
  [docs/licenses/lpips-pytorch.txt](docs/licenses/lpips-pytorch.txt).
  LPIPS weights are downloaded from
  [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity).

`assets/overview.png` is Figure 1 from *Birth-Death Structural Learning for
3D Gaussian Splatting*, by Quan Tran Hong, Long Nguyen-Chi, and Binh T. Nguyen.
