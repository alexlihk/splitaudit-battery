"""
Demo: CE probe on GPT-2/AG News CLS s6 (full-information anchor).
Expected t1 ~ 0.94-0.97. If you get ~0.005 you built the MSE fresher
by mistake (the 444x lesson). ~20 min on one T4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

from probes.probe_common import Head, Tail
from probes.train_cf_attacker import train_cf_attacker
from probes.eval_attacker import eval_attacker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = 6


class ClsHead(nn.Module):
    def __init__(self, dim=768, n=4):
        super().__init__()
        self.fc = nn.Linear(dim, n)

    def forward(self, h, mask):
        idx = torch.arange(h.size(0), device=h.device)
        return self.fc(h[idx, mask.sum(1) - 1])


def main():
    ds = load_dataset("fancyzhx/ag_news")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    def enc(split, n):
        d = ds[split].select(range(n))
        e = tok(list(d["text"]), max_length=64, truncation=True,
                padding="max_length", return_tensors="pt")
        return (e["input_ids"], e["attention_mask"].bool(),
                torch.tensor(list(d["label"])))

    ids_tr, m_tr, y_tr = enc("train", 5000)
    ids_te, m_te, y_te = enc("test", 1000)

    torch.manual_seed(1000 + SPLIT)
    full = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    head = Head(full, SPLIT).to(DEVICE)
    tail = Tail(full, SPLIT).to(DEVICE)
    cls = ClsHead().to(DEVICE)
    del full

    loader = DataLoader(TensorDataset(ids_tr, m_tr, y_tr),
                        batch_size=64, shuffle=True)
    params = list(dict.fromkeys(
        list(head.parameters()) + list(tail.parameters())
        + list(cls.parameters())))
    opt = torch.optim.AdamW(params, lr=2e-5)
    for ep in range(3):
        for ids, mm, yy in loader:
            ids = ids.to(DEVICE)
            mm = mm.to(DEVICE)
            yy = yy.to(DEVICE)
            loss = F.cross_entropy(cls(tail(head(ids)), mm), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for p in params:
        p.requires_grad_(False)
    head.eval()
    tail.eval()
    cls.eval()

    wte = head.wte.weight.detach()
    with torch.no_grad():
        H_TR = torch.cat([head(ids_tr[i:i + 64].to(DEVICE)).cpu()
                          for i in range(0, len(ids_tr), 64)])
        H_TE = torch.cat([head(ids_te[i:i + 64].to(DEVICE)).cpu()
                          for i in range(0, len(ids_te), 64)])

    att = train_cf_attacker(H_TR, ids_tr, m_tr, wte, device=DEVICE)
    r = eval_attacker(att, H_TE, ids_te, m_te, wte, device=DEVICE)
    print("Full-information anchor: t1=%.4f (expected 0.94-0.97) t5=%.4f"
          % (r["t1"], r["t5"]))


if __name__ == "__main__":
    main()
