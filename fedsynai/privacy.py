"""Client-level differential privacy for FL (DP-FedAvg): clip each client update, add Gaussian noise.
RDP accountant for the full-participation Gaussian mechanism (no subsampling amplification)."""
import math
import torch


def clip_and_noise(deltas: list[torch.Tensor], clip: float, noise_mult: float, gen: torch.Generator):
    """deltas: per-client model updates. Returns (noisy mean update, clipped fraction)."""
    clipped, n_clipped = [], 0
    for d in deltas:
        norm = d.norm().item()
        n_clipped += norm > clip
        clipped.append(d * min(1.0, clip / (norm + 1e-12)))
    total = torch.stack(clipped).sum(0)
    if noise_mult > 0:
        total = total + torch.randn(total.shape, generator=gen) * noise_mult * clip
    return total / len(deltas), n_clipped / len(deltas)


def epsilon(noise_mult: float, rounds: int, delta: float = 1e-5) -> float:
    """(eps, delta)-DP after `rounds` Gaussian-mechanism compositions via Renyi DP."""
    if noise_mult <= 0:
        return float("inf")
    best = float("inf")
    for a in [1 + x / 10 for x in range(1, 100)] + list(range(12, 512)):
        rdp = rounds * a / (2 * noise_mult ** 2)
        best = min(best, rdp + math.log(1 / delta) / (a - 1))
    return best
