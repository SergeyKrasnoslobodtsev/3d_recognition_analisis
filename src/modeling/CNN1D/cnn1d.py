import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, groups=1, act=True):
        super().__init__()
        p = (k // 2) * d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                              dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SqueezeExcite1d(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(1, ch // r)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, ch, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class DWSeparableBlock1d(nn.Module):
    def __init__(self, ch, k=5, d=1, drop=0.1, use_se=True):
        super().__init__()
        self.dw = ConvBNAct1d(ch, ch, k=k, d=d, groups=ch)
        self.pw = ConvBNAct1d(ch, ch, k=1)
        self.se = SqueezeExcite1d(ch) if use_se else nn.Identity()
        self.drop = nn.Dropout1d(p=drop) if drop and drop > 0 else nn.Identity()
    def forward(self, x):
        out = self.drop(self.se(self.pw(self.dw(x))))
        return x + out

class DownsampleBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNAct1d(in_ch, out_ch, k=3, s=2)
        self.res  = ConvBNAct1d(in_ch, out_ch, k=1, s=2, act=False)
    def forward(self, x): return self.conv(x) + self.res(x)

class CNN1D_Backbone(nn.Module):
    def __init__(self, in_ch=7, c_stem=32, widths=(64, 128, 256), drop=0.1, use_se=True):
        super().__init__()
        self.stem = ConvBNAct1d(in_ch, c_stem, k=7)
        stages, c_prev = [], c_stem
        for c in widths:
            stages += [DownsampleBlock1d(c_prev, c),
                       DWSeparableBlock1d(c, k=5, d=1, drop=drop, use_se=use_se),
                       DWSeparableBlock1d(c, k=5, d=2, drop=drop, use_se=use_se)]
            c_prev = c
        self.stages = nn.Sequential(*stages)
        self.out_ch = widths[-1]
        self.head_norm = nn.BatchNorm1d(self.out_ch)
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head_norm(x)

class CNN1D(nn.Module):
    def __init__(self, in_ch=7, emb_dim=256, num_classes=0, c_stem=32, widths=(64,128,256), drop=0.1, use_se=True):
        super().__init__()
        self.backbone = CNN1D_Backbone(in_ch, c_stem, widths, drop, use_se)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Sequential(
            nn.Conv1d(self.backbone.out_ch, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.SiLU(inplace=True)
        )
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(emb_dim, self.num_classes) if self.num_classes > 0 else None
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # SiLU совместим по gain с ReLU
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, return_embedding=False):
        f = self.backbone(x)       # (B,C,L')
        f = self.pool(f)           # (B,C,1)
        f = self.embed(f).squeeze(-1)  # (B,emb)
        if self.classifier is None or return_embedding:
            return F.normalize(f, dim=1) if return_embedding else f
        return self.classifier(f)

def build_cnn1d_base(num_classes=0, emb_dim=256):
    return CNN1D(in_ch=7, emb_dim=emb_dim, num_classes=num_classes, c_stem=32, widths=(64,128,256), drop=0.10, use_se=True)