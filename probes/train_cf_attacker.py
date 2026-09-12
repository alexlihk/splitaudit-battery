"""
Tier 2 CE decoder - the primary instrument.
Source: exp2e(v3); used in 10 of 14 experiments.
Cross-session: LM s6 0.9788 vs 0.9820; floors exact.
Patent CN 202611402684.7 claim 5 (tiering), claim 12 (budget).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconNet(nn.Module):
    """MLP 768-2048-2048-768 GELU. Identical capacity for MSE and CE
    probes: the 444x instrument gap is attributable solely to loss."""

    def __init__(self, dim=768, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        return self.net(z)


def ce_token(z, ids, mask, wte):
    D = z.shape[-1]
    zf = z.reshape(-1, D)
    tgt = ids.reshape(-1)
    m = mask.reshape(-1)
    idx = torch.nonzero(m).squeeze(1)
    if idx.numel() == 0:
        return 0.0 * z.sum()
    return F.cross_entropy(zf[idx] @ wte.t(), tgt[idx])


def train_cf_attacker(X, ids, mask, wte, seed=4321, epochs=2,
                      lr=1e-3, bs=32, hidden=2048, device="cuda"):
    torch.manual_seed(seed)
    att = ReconNet(hidden=hidden).to(device)
    opt = torch.optim.Adam(att.parameters(), lr=lr)
    N = X.shape[0]
    for ep in range(epochs):
        att.train()
        perm = torch.randperm(N)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            xb = X[idx].to(device)
            ii = ids[idx].to(device)
            mm = mask[idx].to(device)
            loss = ce_token(att(xb), ii, mm, wte)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for p in att.parameters():
        p.requires_grad_(False)
    att.eval()
    return att
