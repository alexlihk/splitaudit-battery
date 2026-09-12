"""
Shared model skeleton (official GPT2Model API, transformers 4.5x+).
Used verbatim across exp2e/1b/5b/.../14. Sanity: reconstruction
max|delta| = 0.00 in all sessions.
"""
import copy
import torch.nn as nn
from transformers import GPT2Model


def _force_eager(blocks):
    for blk in blocks:
        try:
            blk.attn.attn_implementation = "eager"
        except Exception:
            pass


class Head(nn.Module):
    """Client: wte/wpe/drop + layers 0..s-1 (official API)."""

    def __init__(self, full, s):
        super().__init__()
        cfg = copy.deepcopy(full.config)
        cfg.n_layer = s
        self.core = GPT2Model(cfg)
        self.core.h = nn.ModuleList(list(full.transformer.h)[:s])
        self.core.wte = full.transformer.wte
        self.core.wpe = full.transformer.wpe
        self.core.ln_f = nn.Identity()
        _force_eager(self.core.h)

    @property
    def wte(self):
        return self.core.wte

    def forward(self, ids):
        return self.core(input_ids=ids, use_cache=False).last_hidden_state


class Tail(nn.Module):
    """Server: layers s..11 + ln_f. inputs_embeds path; wpe zeroed."""

    def __init__(self, full, s):
        super().__init__()
        cfg = copy.deepcopy(full.config)
        cfg.n_layer = 12 - s
        self.core = GPT2Model(cfg)
        self.core.h = nn.ModuleList(list(full.transformer.h)[s:])
        self.core.ln_f = full.transformer.ln_f
        self.core.wte = full.transformer.wte
        self.core.wpe.weight.data.zero_()
        self.core.wpe.weight.requires_grad_(False)
        self.core.drop = nn.Identity()
        _force_eager(self.core.h)

    def forward(self, h):
        return self.core(inputs_embeds=h, use_cache=False).last_hidden_state
