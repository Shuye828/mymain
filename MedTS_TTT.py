import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def ln_fwd(x, gamma, beta, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    # gradient of 0.5*||LN(x)-l2_target||^2 wrt x (batch-wise)
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_out = (y - l2_target)          # dL/dy
    grad_xhat = grad_out * gamma        # dL/dx_hat
    z = (1.0 / D) * (D * grad_xhat - grad_xhat.sum(-1, True) - x_hat * (grad_xhat * x_hat).sum(-1, True))
    return z / std

class LayerNorm1D(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        mean = x.mean(dim=1, keepdim=True)
        var  = (x - mean).to(torch.float32).pow(2).mean(dim=1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var.to(dtype) + self.eps)
        return x_hat * self.weight[:, None] + self.bias[:, None]

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class TTTLinear(nn.Module):
    def __init__(self, dim, num_heads, base_lr=1.0, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.base_lr = base_lr
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # slow weights (learned by outer-loop optimizer)
        self.W = nn.Parameter(torch.randn(1, num_heads, self.head_dim, self.head_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))

        # per-head LN params (slow)
        self.ln_w = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim))
        self.ln_b = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))

        # optional: learn per-head lr gate from sample mean feature (slow)
        self.lr_w = nn.Parameter(torch.randn(num_heads, dim) * 0.02)
        self.lr_b = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d).transpose(1, 2)

        ##---------------------------------------------inner train------------------------------------------------------
        W = self.W.expand(B, -1, -1, -1).contiguous()
        b = self.b.expand(B, -1, -1, -1).contiguous()
        Z = k @ W + b
        target = v - k
        G = ln_fused_l2_bwd(Z, target, self.ln_w, self.ln_b, eps=self.eps)
        x_mean = x.mean(dim=1)
        lr = torch.sigmoid(torch.einsum("bd,hd->bh", x_mean, self.lr_w) + self.lr_b)
        eta = (self.base_lr / float(d)) * lr.view(B, H, 1, 1)
        # fast weight step
        W = W - eta * (k.transpose(-2, -1) @ G)
        b = b - eta * G.sum(dim=-2, keepdim=True)
        ##--------------------------------------------------------------------------------------------------------------
        Zq = q @ W + b
        Zq = ln_fwd(Zq, self.ln_w, self.ln_b, eps=self.eps)
        y = q + Zq
        y = y.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(y)

class MedTSTTTLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MedTSTTTLayer, self).__init__()
        hidden_size = dim * 2
        self.norm = RMSNorm(dim)
        self.norm_Gate = RMSNorm(hidden_size)
        self.in_proj = nn.Linear(dim, 2 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, dim, bias=False)
        self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, groups=hidden_size, kernel_size=4, padding=3)
        self.ttt = TTTLinear(dim=hidden_size, num_heads=num_heads)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x, res = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))
        x = F.silu(x[:, :, :shortcut.shape[1]].transpose(1, 2))
        x = self.ttt(x)
        x = F.silu(res) * x
        x = self.out_proj(self.norm_Gate(x)) + shortcut
        return x

class MedTSTTT(nn.Module):
    def __init__(self, dim, max_channel=128, num_heads=8, num_layers=8, patch_size=16, num_classes=5):
        super(MedTSTTT, self).__init__()
        self.patch_size = patch_size
        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=(1, 25), padding=(0, 12))
        self.spatial_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(max_channel, 1))
        self.norm = LayerNorm1D(dim)
        self.proj = nn.Conv1d(in_channels=dim, out_channels=dim, bias=False, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, dim), requires_grad=True)
        self.layers = nn.ModuleList([
            MedTSTTTLayer(dim=dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.classification_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b, c, l = x.shape
        # ----------------------------------------Z-Score---------------------------------------------------------------
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)
        # --------------------------------------------------------------------------------------------------------------
        remainder = self.patch_size * math.ceil(l / self.patch_size) - l
        if remainder > 0:
            x = F.pad(x, (0, remainder), mode='constant', value=0)
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = F.conv2d(x, self.spatial_conv.weight[:, :, 0:c, :])
        x = x.squeeze(2)
        x = F.gelu(self.norm(x))
        x = self.proj(x).permute(0, 2, 1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, dim=1, keepdim=False)
        pre = self.classification_head(x)
        return pre


if __name__ == "__main__":
    x = torch.randn(3, 10, 256)
    model = MedTSTTT(dim=128, num_classes=5)
    y = model(x)
    print(y.shape)
