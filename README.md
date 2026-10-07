# splitaudit-battery — README v2（補丁版，整檔替換 repo 現有 README.md）

# SplitAudit Battery (L1: calibration & probes)

Calibrated probe battery for auditing representation leakage in
split-learning / split-inference deployments. Companion artifact to:
"Split-Learning and Split-Inference Privacy Defenses Fail Under Strong
Attackers: A Metric-Agnostic Dual-Track Empirical Audit of
Representation Leakage" (preprint DOI: 10.13140/RG.2.2.32481.67681)
and patent application CN 202611402684.7.

## Calibration anchors (verify your fork)

Anchors are model x corpus x family dependent. Reference points
(GPT-2 / AG News unless noted):

| Anchor | Value | Meaning |
|---|---|---|
| Full information | 0.97 top-1 (124M); 0.957 (355M); 0.907 (774M, 5-config mean, exp18b); 0.934 (Llama-3.2-1B, exp19) | unmodified CLS s6 representations |
| Ambient baseline | 0.317 (124M); 0.356 (355M); 0.374 (774M); 0.329 (DBpedia); 0.415 (Llama-3.2-1B); 0.466 (Llama-3.1-8B, exp23b); 0.522 (Mistral-7B, exp21b) | pretrained LM prior, no access to z (P_ctx) |
| Zero information | 0.030 | collapsed codebook (unigram mode floor) |
| Two-probe closure | 0.469 ~= 0.475 | reassembler vs SeqDecoder on 9-bit |

Any fork whose numbers do not reproduce these anchors is not
calibrated. Note (exp18b): probe training carries ~7pp run-to-run
variance — multi-seed averaging is mandatory; single-seed readouts
with a large t5-t1 gap are low draws, not capacity limits.

## Quick start

    pip install torch transformers datasets matplotlib
    python probes/demo.py            # full-info anchor (~20 min T4)
    python calibration/anchors.py    # ambient + zero (~15 min T4)
    python figs/figs_all.py          # all figures (CPU, 10 s)

## Contents

    probes/       Tier 0-3 probes (floor / CE / reassembler)
    vq/           repaired VQ cell (k-means init, EMA, revive)
    calibration/  anchor fixture
    results/      18 paper-backed JSONs (exp1b-19) + 9 H20 JSONs (exp19b, exp20-23, exp21b/exp23b 7B+ probe recalibration)
    figs/         figure scripts (hardcoded measured values)

## Provenance

Cross-session reproduction: LM s6 0.9788 vs 0.9820; floors match to
all printed digits (0.0055 / 0.0029). exp18b control: same 774M
representations x 5 probe configs -> t1 0.888-0.914 (mean 0.907).
exp19: Llama-3.2-1B cross-family confirmation (t1 0.9341, t5 0.9805,
P_ctx 0.4148).
exp21b/exp23b (2026-10-07): 7B+ probe re-calibration, 5 candidates x 3 seeds at
registered budget - 0/5 anchor separation (threshold t1 > P_ctx + 0.15; best gaps:
Mistral +0.134, Llama-8B +0.052). 7B+ remains REVIEW (instrument boundary,
fail-closed). Caveat: v1 script seed wiring made all seeds identical (t1_std=0.0)
- treat these as single-seed reads; script fixed in splitrisk commit 3bc21376.
Per-file headers carry the experiment ledger.

## License

BSD-3-Clause (language layer). Engine layer (certification,
subscription, threshold library) is not part of this repository.

## Citation

See CITATION.cff. Repository: https://github.com/alexlihk/splitaudit-battery
Product: https://github.com/alexlihk/splitrisk (SplitRisk, L1 CLI package)
