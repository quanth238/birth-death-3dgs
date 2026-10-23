# Vanilla transfer results

Results use the seven-scene Mip-NeRF360 subset listed in the README.

Both methods use SfM initialization, seed 0, 7,000 iterations, the standard
held-out split, and Vanilla clone/split/prune operations. ADC starts after
iteration 500 and runs every 100 iterations.

First variation uses 16 score views, `0.8 * L1 + 0.2 * (1 - SSIM)`, and no EMA
or clipping. Parents must satisfy `q_i > 1e-7`; each event adds at most 10% of
the current population. If more parents qualify, we sample without replacement
in proportion to `q_i - 1e-7`. There is no final Gaussian cap. This setting was
selected on Bicycle and applied unchanged to the other six scenes in this table.

| Scene | Vanilla PSNR | First variation PSNR | Δ PSNR | Δ SSIM | LPIPS reduction | Vanilla Gaussians | First variation Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bicycle | 23.5719 | 23.9188 | +0.3469 | +0.0381 | +0.0527 | 2,988,156 | 1,792,720 |
| Bonsai | 30.0848 | 30.3209 | +0.2361 | +0.0077 | +0.0262 | 1,041,078 | 1,673,656 |
| Counter | 27.3214 | 27.7344 | +0.4130 | +0.0094 | +0.0249 | 954,473 | 1,709,618 |
| Garden | 26.1611 | 26.1721 | +0.0110 | +0.0029 | +0.0061 | 3,591,833 | 1,938,234 |
| Kitchen | 29.3233 | 29.7719 | +0.4486 | +0.0055 | +0.0126 | 1,436,371 | 2,123,917 |
| Room | 29.5543 | 30.0436 | +0.4894 | +0.0078 | +0.0207 | 1,014,529 | 1,399,168 |
| Stump | 25.5922 | 25.9889 | +0.3967 | +0.0314 | +0.0457 | 3,250,050 | 1,520,958 |

Means give each scene equal weight. Full-precision metrics, Gaussian counts,
and per-view results, including the two additional scene runs, are in
[results.json](results.json). Runs used an RTX 4090, PyTorch 2.11.0, and CUDA 12.8.
