import torch
from dataclasses import dataclass

@dataclass
class SpecAugConfig:
    specaug_frac: float = 0.10
    specaug_slices: int = 2
    channel_drop_p: float = 0.05

def augment_batch_matrix(xb: torch.Tensor, cfg: SpecAugConfig):
    # xb: (B, 7, L)
    B, C, L = xb.shape
    if cfg.channel_drop_p > 0:
        mask = torch.rand(B, device=xb.device) < cfg.channel_drop_p
        if mask.any():
            drop_c = torch.randint(0, C, (int(mask.sum().item()),), device=xb.device)
            xb[mask, drop_c, :] = 0.0
    if cfg.specaug_frac > 0 and cfg.specaug_slices > 0:
        w = max(1, int(round(L * cfg.specaug_frac)))
        for b in range(B):
            for _ in range(cfg.specaug_slices):
                s = torch.randint(0, max(1, L - w + 1), (1,), device=xb.device).item()
                xb[b, :, s:s+w] = 0.0
    return xb