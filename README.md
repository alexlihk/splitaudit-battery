# SplitAudit Battery (L1: calibration & probes)

Calibrated probe battery for auditing representation leakage in
split-learning / split-inference deployments. Companion artifact to:
"Split-Learning and Split-Inference Privacy Defenses Fail Under Strong
Attackers: A Metric-Agnostic Dual-Track Empirical Audit" (arXiv 2026 submit/8072557 )
and patent application CN 202611402684.7.

## Calibration anchors (verify your fork)

| Anchor | Value | Meaning |
|---|---|---|
| Full information | 0.97 top-1 | unmodified CLS s6 representations |
| Ambient baseline | 0.317 | pretrained GPT-2, no access to z (P_ctx) |
| Zero information | 0.030 | collapsed codebook (unigram mode floor) |
| Two-probe closure | 0.469 ~= 0.475 | reassembler vs SeqDecoder on 9-bit |

Any fork whose numbers do not reproduce these anchors is not calibrated.

## Quick start

    pip install torch transformers datasets matplotlib
    python probes/demo.py            # full-info anchor (~20 min T4)
    python calibration/anchors.py    # ambient + zero (~15 min T4)
    python figs/figs_all.py          # all figures (CPU, 10 s)

## Contents

    probes/       Tier 0-3 probes (floor / CE / reassembler)
    vq/           repaired VQ cell (k-means init, EMA, revive)
    calibration/  anchor fixture
    results/      12 experiment JSONs (fixed seeds)
    figs/         figure scripts (hardcoded measured values)

## Provenance

Cross-session reproduction: LM s6 0.9788 vs 0.9820; floors match to all
printed digits (0.0055 / 0.0029). The CE probe is the primary instrument
in 10 of 14 experiments. Per-file headers carry the experiment ledger.

## License

BSD-3-Clause (language layer). Engine layer (certification, subscription,
threshold library) is not part of this repository.

## Citation

See CITATION.cff. Repository: REPLACE-WITH-FINAL-URL
