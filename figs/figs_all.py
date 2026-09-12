"""
All paper figures, hardcoded from measured values (results/*.json).
CPU-only, matplotlib only. Values frozen at arXiv submission.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

D = [2, 4, 6, 8, 10]
LM_MSE = [0.0545, 0.0218, 0.0070, 0.0022, 0.0154]
LM_CE = [0.9989, 0.9836, 0.9820, 0.9758, 0.9600]
CLS_CE = [0.9923, 0.9771, 0.9757, 0.9678, 0.9556]
LM_FLR = [0.0687, 0.0254, 0.0077, 0.0055, 0.0153]
EXP1 = [0.0961, 0.0267, 0.0095, 0.0032, 0.0147]
VQ_BITS = [4, 8, 12, 16]
VQ_T1 = [0.0101, 0.3059, 0.5398, 0.7277]
VQ_ORC = [0.0530, 0.3192, 0.5570, 0.7468]
VQ_PPLX = [40.92, 11.19, 4.55, 2.78]
NSH = [3, 5, 10, 20]
P1_T1 = [0.923, 0.878, 0.775, 0.602]

# fig2: weak vs strong
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(D, LM_CE, "o-", c="#d62728", label="CE probe (strong)")
ax.plot(D, LM_MSE, "o--", c="#1f77b4", label="MSE fresher (weak)")
ax.plot(D, LM_FLR, "o:", c="#2ca02c", label="Floor (Tier 0)")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1.05)
ax.annotate("444x gap (layer 8)", xy=(8, 0.0022), xytext=(3.5, 0.006),
            fontsize=9, arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Split depth")
ax.set_ylabel("Top-1 recovery (log)")
ax.set_title("The attacker loss function sets the measurement")
ax.legend()
fig.savefig(OUT + "/fig2.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# fig5: phase diagram
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.plot(D, LM_CE, "o-", c="#d62728", label="LM CE (true)")
ax.plot(D, CLS_CE, "s-", c="#d62728", alpha=0.55, label="CLS CE (true)")
ax.plot(D, LM_FLR, "o:", c="#2ca02c", label="LM floor")
ax.plot(D, LM_MSE, "o--", c="#1f77b4", label="LM MSE (=exp1 instrument)")
ax.plot(D, EXP1, "x-", c="#555555", label="exp1 published claim")
ax.set_yscale("log")
ax.set_ylim(1e-3, 1.05)
ax.annotate("depth removes linear readability (floor 14x)\n"
            "not learned decodability (CE flat)",
            xy=(8, 0.0055), xytext=(2.6, 0.0025), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Split depth")
ax.set_ylabel("Top-1 recovery (log)")
ax.set_title("Corrected phase diagram")
ax.legend(fontsize=8)
fig.savefig(OUT + "/fig5.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# fig4: VQ capacity
fig, ax = plt.subplots(figsize=(8, 5.5))
ax2 = ax.twinx()
ax.plot(VQ_BITS, VQ_T1, "o-", c="#d62728", label="recovery (CE probe)")
ax.plot(VQ_BITS, VQ_ORC, "o--", c="#7f8c8d", label="oracle bound")
ax2.plot(VQ_BITS, VQ_PPLX, "s-", c="#1f77b4", label="PPL ratio (right)")
ax2.set_yscale("log")
ax.annotate("16 bits = raw token ID info", xy=(16, 0.7277),
            xytext=(9.5, 0.80), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("VQ bits per position (log2 K)")
ax.set_ylabel("Top-1 recovery")
ax2.set_ylabel("PPL ratio (log)")
ax.set_title("Task capacity = attack capacity")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8)
fig.savefig(OUT + "/fig4.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# fig6: sharding
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(NSH, P1_T1, "o-", c="#1f77b4", label="single-shard view")
axes[0].axhline(0.91, ls="--", c="#d62728", label="collusive ceiling 0.91")
axes[0].set_xlabel("Number of shards N")
axes[0].set_ylabel("Top-1 recovery")
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title("(a) N lowers single view, not the ceiling")
axes[0].legend(fontsize=8)
labels = ["single\n0.41", "k=1\n0.434", "k=2\n0.449",
          "k=3\n0.469", "unsharded\n0.475"]
vals = [0.410, 0.434, 0.449, 0.469, 0.475]
axes[1].bar(range(5), vals,
            color=["#e67e22", "#e67e22", "#e67e22", "#d62728", "#7f8c8d"])
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(labels, fontsize=8)
axes[1].set_ylim(0, 0.6)
axes[1].set_ylabel("Top-1 (9-bit)")
axes[1].set_title("(b) 9-bit collusion ladder")
fig.suptitle("Sharding is orthogonal: bits set the ceiling", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT + "/fig6.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Figures written to " + OUT)
