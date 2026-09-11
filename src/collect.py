"""Collect every run's result.json, write a summary table and the figures."""
import json, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("RUNS", "runs")
OUT = os.environ.get("OUT", "figures")
os.makedirs(OUT, exist_ok=True)

rows = []
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    a, fin = r.get("args", {}), r.get("final", {})
    if not fin:
        continue
    rows.append({"name": os.path.basename(os.path.dirname(f)), "args": a, "final": fin})

print(f"collected {len(rows)} runs\n")


def agg(pred):
    """Mean/sd per test split over seeds matching pred."""
    sel = [r for r in rows if pred(r["args"], r["name"])]
    if not sel:
        return None
    keys = sorted(set(k for r in sel for k in r["final"]),
                  key=lambda k: int(re.sub(r"\D", "", k) or 0))
    return keys, {k: (float(np.mean([r["final"].get(k, 0) for r in sel])),
                      float(np.std([r["final"].get(k, 0) for r in sel])),
                      len(sel)) for k in keys}


# ---- Figure 1: the method vs the baseline -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
colors = {0: "#c0392b", 1: "#1f77b4"}
for ax, mode in zip(axes, ["ar", "diff"]):
    for coup in (0, 1):
        got = agg(lambda a, n, c=coup, m=mode: a.get("coupled") == c and a.get("mode") == m
                  and a.get("pe") == "alibi" and n.startswith("coup-"))
        if not got:
            continue
        keys, st = got
        xs = [int(re.sub(r"\D", "", k)) for k in keys]
        mu = [st[k][0] * 100 for k in keys]
        sd = [st[k][1] * 100 for k in keys]
        lbl = "significance-aligned (ours)" if coup else "sequential (baseline)"
        ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3, color=colors[coup], label=lbl)
    ax.axvline(5.5, ls="--", c="gray", lw=1)
    ax.text(5.6, 92, "trained\n≤ 5 digits", fontsize=8, color="gray")
    ax.set_title({"ar": "Autoregressive", "diff": "Masked diffusion"}[mode])
    ax.set_xlabel("test operand digits")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=.3)
axes[0].set_ylabel("exact-match accuracy (%)")
axes[0].legend(fontsize=8)
fig.suptitle("Length generalization on addition (3 seeds, mean ± sd)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_method_vs_baseline.png", dpi=160)
print("wrote fig1_method_vs_baseline.png")

# ---- Figure 2: positional encoding transfers asymmetrically -----------------
pes = ["nope", "ape", "sin", "alibi"]
lbl = {"nope": "none", "ape": "learned\nabsolute", "sin": "sinusoidal", "alibi": "distance\nrule"}
fig, ax = plt.subplots(figsize=(7, 4.2))
w = 0.36
for i, mode in enumerate(["ar", "diff"]):
    mus, sds = [], []
    for pe in pes:
        got = agg(lambda a, n, p=pe, m=mode: a.get("pe") == p and a.get("mode") == m
                  and n.startswith(("add-", "seed-")))
        if got:
            keys, st = got
            k = "id" if "id" in st else keys[0]
            mus.append(st[k][0] * 100); sds.append(st[k][1] * 100)
        else:
            mus.append(0); sds.append(0)
    ax.bar(np.arange(len(pes)) + (i - .5) * w, mus, w, yerr=sds, capsize=3,
           label={"ar": "Autoregressive", "diff": "Masked diffusion"}[mode],
           color=["#7f8c8d", "#e67e22"][i])
ax.set_xticks(range(len(pes)))
ax.set_xticklabels([lbl[p] for p in pes])
ax.set_ylabel("in-distribution accuracy (%)")
ax.set_title("Same encoding, opposite outcome: schemes that work for\n"
             "autoregressive models leave diffusion unable to learn")
ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_pe_asymmetry.png", dpi=160)
print("wrote fig2_pe_asymmetry.png")

# ---- summary table ----------------------------------------------------------
lines = ["| run group | " + " | ".join(f"d{d}" for d in [5, 6, 7, 8, 10, 12]) + " | seeds |",
         "|---|" + "---|" * 7]
for coup in (0, 1):
    for mode in ("ar", "diff"):
        for pe in ("ape", "alibi"):
            got = agg(lambda a, n, c=coup, m=mode, p=pe: a.get("coupled") == c
                      and a.get("mode") == m and a.get("pe") == p and n.startswith("coup-"))
            if not got:
                continue
            keys, st = got
            nm = f"{'ours' if coup else 'baseline'} / {mode} / {pe}"
            cells = []
            for d in [5, 6, 7, 8, 10, 12]:
                k = f"d{d}"
                cells.append(f"{st[k][0]*100:.1f}±{st[k][1]*100:.0f}" if k in st else "-")
            lines.append(f"| {nm} | " + " | ".join(cells) + f" | {st[keys[0]][2]} |")
open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
