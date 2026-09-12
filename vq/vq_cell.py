"""
Repaired VQ cell (exp5b): k-means init from data, EMA, dead-code revive.
Frontier: 4-bit 1%/41x -> 16-bit 73%/2.8x; oracle 0.747.
Bitrate is a measured parameter/decision variable, never a privacy step.
Patent CN 202611402684.7 claims 7/12.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, dim=768, K=512, ema=0.99, eps=1e-5):
        super().__init__()
        self.K, self.ema, self.eps = K, ema, eps
        self.register_buffer("codebook", torch.zeros(K, dim))
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_sum", torch.zeros(K, dim))
        self.register_buffer("usage_acc", torch.zeros(K))

    @torch.no_grad()
    def init_from_data(self, sample):
        M = sample.shape[0]
        sub = sample[:100000]
        if self.K <= 4096 and M >= self.K:
            idx0 = torch.randperm(M, device=sample.device)[:self.K]
            cb = sample[idx0].clone()
            for _ in range(8):
                a = self._assign(cb, sub)
                cnt = torch.zeros(self.K, device=sample.device)
                cnt.scatter_add_(0, a, torch.ones_like(a, dtype=torch.float))
                sm = torch.zeros(self.K, sample.shape[1],
                                 device=sample.device)
                sm.scatter_add_(0, a.unsqueeze(1)
                                .expand(-1, sample.shape[1]), sub)
                ok = cnt > 0
                cb[ok] = sm[ok] / cnt[ok].unsqueeze(1)
        else:
            idx0 = torch.randint(0, M, (self.K,), device=sample.device)
            cb = sample[idx0].clone()
        self.codebook.copy_(cb)
        self.cluster_size.fill_(1.0)
        self.embed_sum.copy_(cb)
        self.usage_acc.zero_()

    def _assign(self, cb, z):
        cbn = (cb ** 2).sum(1)
        out = []
        for i in range(0, z.shape[0], 1024):
            zc = z[i:i + 1024]
            d = (zc.pow(2).sum(1, keepdim=True) + cbn.unsqueeze(0)
                 - 2.0 * (zc @ cb.t()))
            out.append(d.argmin(1))
        return torch.cat(out)

    def forward(self, z_e, training=False):
        B, T, D = z_e.shape
        z = z_e.reshape(-1, D)
        with torch.no_grad():
            idx = self._assign(self.codebook, z)
            z_q = self.codebook[idx].reshape(B, T, D)
        z_q_st = z_e + (z_q - z_e).detach()
        if training:
            with torch.no_grad():
                cnt = torch.zeros(self.K, device=z.device)
                cnt.scatter_add_(0, idx,
                                 torch.ones_like(idx, dtype=torch.float))
                sm = torch.zeros(self.K, D, device=z.device)
                sm.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), z)
                self.cluster_size.mul_(self.ema).add_(cnt, alpha=1 - self.ema)
                self.embed_sum.mul_(self.ema).add_(sm, alpha=1 - self.ema)
                self.codebook.copy_(
                    self.embed_sum
                    / self.cluster_size.clamp(min=self.eps).unsqueeze(1))
                self.usage_acc.add_(cnt)
        return z_q_st, F.mse_loss(z_e, z_q.detach()), idx.reshape(B, T), 0.0

    @torch.no_grad()
    def revive(self, sample):
        dead = self.usage_acc < 0.5
        nd = int(dead.sum())
        if nd > 0 and sample.shape[0] > 0:
            ii = torch.randint(0, sample.shape[0], (nd,),
                               device=sample.device)
            self.codebook[dead] = sample[ii]
            self.cluster_size[dead] = 1.0
            self.embed_sum[dead] = sample[ii]
        self.usage_acc.zero_()
