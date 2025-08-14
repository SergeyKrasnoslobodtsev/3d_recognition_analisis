import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 2, drop: float = 0.2):
        super().__init__()
        hid = dim * mult
        self.fc1 = nn.Linear(dim, hid, bias=False)
        self.bn1 = nn.BatchNorm1d(hid)
        self.fc2 = nn.Linear(hid, dim, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        y = self.drop(F.silu(self.bn1(self.fc1(x))))
        y = self.bn2(self.fc2(y))
        return F.silu(x + y)

class PooledMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, width: int = 256, depth: int = 2, drop: float = 0.2):
        super().__init__()
        self.in_norm = nn.BatchNorm1d(in_dim)
        self.in_proj = nn.Linear(in_dim, width, bias=False)
        self.in_bn   = nn.BatchNorm1d(width)
        self.in_act  = nn.SiLU(inplace=True)
        self.in_drop = nn.Dropout(drop)
        self.blocks  = nn.Sequential(*[ResidualMLPBlock(width, mult=2, drop=drop) for _ in range(depth)])
        self.head    = nn.Sequential(nn.Linear(width, width, bias=False), nn.BatchNorm1d(width), nn.SiLU(inplace=True))
        self.classifier = nn.Linear(width, num_classes)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):  # (B, D)
        x = self.in_drop(self.in_act(self.in_bn(self.in_proj(self.in_norm(x)))))
        x = self.blocks(x)
        x = self.head(x)
        return self.classifier(x)