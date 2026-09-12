"""
Tier 0 linear floor - zero-parameter cosine alignment.
Depth series 0.069->0.005 (14x) with CE flat = "depth buys
obfuscation, not privacy" (exp1b).
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def raw_floor(X, ids, mask, wte, bs=32, device="cuda"):
    wte_n = F.normalize(wte, dim=-1)
    t1 = t5 = n = 0
    for s in range(0, X.shape[0], bs):
        xb = X[s:s + bs].to(device)
        ii = ids[s:s + bs].to(device)
        mm = mask[s:s + bs].to(device)
        D = xb.shape[-1]
        top = (F.normalize(xb.reshape(-1, D), dim=-1)
               @ wte_n.t()).topk(5, -1).indices
        tgt = ii.reshape(-1)
        m = mm.reshape(-1)
        t1 += ((top[:, 0] == tgt) & m).sum().item()
        t5 += ((top == tgt.unsqueeze(1)).any(1) & m).sum().item()
        n += m.sum().item()
    return t1 / max(n, 1), t5 / max(n, 1)
