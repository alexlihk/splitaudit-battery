"""
Tier 3 collusion reassembler (exp10b/11).
Unquantized all-shard 0.91; 9-bit all-shard 0.469 (closure 0.475).
Patent CN 202611402684.7 claim 11 (collusion enumeration).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reassembler(nn.Module):
    def __init__(self, vocab=50257, max_len=64, dim=256, layers=3, heads=4):
        super().__init__()
        self.proj = nn.Linear(768, dim)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.1, batch_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(dim, vocab)

    def forward(self, h_sub, pos):
        z = self.proj(h_sub) + self.pos(pos)
        return self.out(self.enc(z))


def train_reassembler(X_tr, P_tr, Y_tr, M_tr, vocab=50257, max_len=64,
                      seed=6000, epochs=6, bs=16, lr=3e-4, device="cuda"):
    torch.manual_seed(seed)
    rec = Reassembler(vocab=vocab, max_len=max_len).to(device)
    opt = torch.optim.AdamW(rec.parameters(), lr=lr)
    N = X_tr.shape[0]
    for ep in range(epochs):
        rec.train()
        perm = torch.randperm(N)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            logits = rec(X_tr[idx].to(device), P_tr[idx].to(device))
            V = logits.size(-1)
            lf = logits.reshape(-1, V)
            tgt = Y_tr[idx].reshape(-1).to(device)
            mm = M_tr[idx].reshape(-1).to(device)
            sel = torch.nonzero(mm).squeeze(1)
            if sel.numel() == 0:
                continue
            loss = F.cross_entropy(lf[sel], tgt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
    rec.eval()
    return rec
