"""
4-anchor calibration fixture.
  full    0.97   exp2e/exp1b base rows (see probes/demo.py)
  ambient 0.317  exp5b3 P_ctx (raw GPT-2, no codes)
  zero    0.030  collapsed codebook, force K=2 (exp5 mode floor)
  closure 0.469 ~= 0.475  exp11 reassembler vs exp5b3B SeqDecoder
A fork that does not separate these anchors is not measuring leakage.
"""
import torch
from transformers import GPT2LMHeadModel

from probes.train_cf_attacker import train_cf_attacker
from probes.eval_attacker import eval_attacker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def anchor_ambient(ids_te, m_te):
    """P_ctx: pretrained GPT-2 next-token prediction, no representation
    access. The leakage an attacker has WITHOUT your deployment."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    t1 = n = 0
    with torch.no_grad():
        for i in range(0, len(ids_te), 32):
            b = ids_te[i:i + 32].to(DEVICE)
            mm = m_te[i:i + 32].to(DEVICE)
            sl = model(input_ids=b).logits[:, :-1]
            V = sl.size(-1)
            top = sl.reshape(-1, V).topk(1, -1).indices[:, 0]
            tgt = b[:, 1:].reshape(-1)
            mk = mm[:, 1:].reshape(-1)
            t1 += ((top == tgt) & mk).sum().item()
            n += mk.sum().item()
    return t1 / max(n, 1)


def anchor_zero(X, ids, mask, wte, K=2):
    """Mode floor: collapsed codebook. Expected ~0.030."""
    from vq.vq_cell import VectorQuantizer
    vq = VectorQuantizer(K=K).to(DEVICE)
    idx = torch.randint(0, X.shape[0] * X.shape[1], (10000,))
    sample = X.reshape(-1, X.shape[-1])[idx].to(DEVICE)
    vq.init_from_data(sample)
    with torch.no_grad():
        Z = torch.cat([vq(X[i:i + 64].to(DEVICE),
                          training=False)[0].cpu()
                       for i in range(0, X.shape[0], 64)])
    att = train_cf_attacker(Z, ids, mask, wte, device=DEVICE)
    return eval_attacker(att, Z, ids, mask, wte,
                         device=DEVICE)["t1"]


if __name__ == "__main__":
    print("Run probes/demo.py first (full-info anchor), then:")
    print("  anchor_ambient  expected 0.317")
    print("  anchor_zero     expected 0.030")
    print("  closure         0.469 ~= 0.475 (9-bit dual decoder)")
