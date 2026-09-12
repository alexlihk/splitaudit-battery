"""
Cosine-recovery evaluation - the honest metric.
Gameability (13 experiments): MSE gameable, CE gameable (norm),
cosine recovery not gameable.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_attacker(att, X, ids, mask, wte, bs=32, device="cuda"):
    wte_n = F.normalize(wte, dim=-1)
    t1 = t5 = n = 0
    ce_sum = 0.0
    for s in range(0, X.shape[0], bs):
        xb = X[s:s + bs].to(device)
        ii = ids[s:s + bs].to(device)
        mm = mask[s:s + bs].to(device)
        z = att(xb)
        D = z.shape[-1]
        zf = z.reshape(-1, D)
        tgt = ii.reshape(-1)
        m = mm.reshape(-1)
        idx = torch.nonzero(m).squeeze(1)
        if idx.numel() > 0:
            ce_sum += (F.cross_entropy(zf[idx] @ wte.t(), tgt[idx]).item()
                       * idx.numel())
        top = (F.normalize(zf, dim=-1) @ wte_n.t()).topk(5, -1).indices
        t1 += ((top[:, 0] == tgt) & m).sum().item()
        t5 += ((top == tgt.unsqueeze(1)).any(1) & m).sum().item()
        n += m.sum().item()
    return {"t1": t1 / max(n, 1), "t5": t5 / max(n, 1),
            "ce": ce_sum / max(n, 1)}
