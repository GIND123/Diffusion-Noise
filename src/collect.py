"""Aggregate every run into tables and figures for the paper."""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("RUNS", "runs")
OUT = os.environ.get("OUT", "figures")
os.makedirs(OUT, exist_ok=True)

R = []
for f in sorted(glob.glob(f"{RUNS}/*/result.json")):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if r.get("final"):
        r["name"] = os.path.basename(os.path.dirname(f))
        R.append(r)
print(f"collected {len(R)} runs")

PRETTY = {"nope": "none", "ape": "learned abs.", "sin": "sinusoidal",
          "rope": "rotary", "alibi": "distance rule"}
ARCH = {"ar": "Autoregressive", "diff": "Masked diffusion"}


def sel(**kw):
    out = []
    for r in R:
        a = r["args"]
        if all(a.get(k) == v for k, v in kw.items() if k != "prefix"):
            if "prefix" not in kw or r["name"].startswith(kw["prefix"]):
                out.append(r)
    return out


def stat(rs, key="final"):
    """mean/sd per digit-length over seeds."""
    if not rs:
        return None
    ds = sorted({int(k[1:]) for r in rs for k in r[key]})
    return ds, {d: (np.mean([r[key].get(f"d{d}", 0) for r in rs]) * 100,
                    np.std([r[key].get(f"d{d}", 0) for r in rs]) * 100, len(rs)) for d in ds}


def envelope(rs, thr):
    """Hardness envelope: largest length still above `thr` accuracy."""
    g = stat(rs)
    if not g:
        return None
    ds, st = g
    ok = [d for d in ds if st[d][0] >= thr * 100]
    return max(ok) if ok else 0


md = ["# Results\n"]

# ---------- Figure 1 + Table 1: the method vs the baseline -------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, mode in zip(axes, ["ar", "diff"]):
    for coup, col, lab in [(0, "#c0392b", "sequential ids (baseline)"),
                           (1, "#1f77b4", "place-value ids (ours)")]:
        g = stat(sel(prefix="A-main", coupled=coup, mode=mode, pe="alibi"))
        if not g:
            continue
        ds, st = g
        ax.errorbar(ds, [st[d][0] for d in ds], yerr=[st[d][1] for d in ds],
                    marker="o", capsize=3, color=col, label=lab)
    ax.axvline(5.5, ls="--", c="gray", lw=1)
    ax.text(5.7, 90, "trained\n≤5 digits", fontsize=8, color="gray")
    ax.set_title(ARCH[mode]); ax.set_xlabel("test operand digits")
    ax.grid(alpha=.3); ax.set_ylim(-3, 103)
axes[0].set_ylabel("exact-match accuracy (%)"); axes[0].legend(fontsize=8)
fig.suptitle("Length generalization on addition (3 seeds, mean ± sd)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_main.png", dpi=160)

md.append("## Table 1 — Main result (addition, distance-rule encoding)\n")
md.append("| configuration | " + " | ".join(f"{d}d" for d in [5, 6, 7, 8, 10, 12, 15, 20]) + " | H*(50%) |")
md.append("|---" * 10 + "|")
for mode in ("ar", "diff"):
    for coup in (0, 1):
        rs = sel(prefix="A-main", coupled=coup, mode=mode, pe="alibi")
        g = stat(rs)
        if not g:
            continue
        _, st = g
        cells = [f"{st[d][0]:.1f}±{st[d][1]:.0f}" if d in st else "–" for d in [5, 6, 7, 8, 10, 12, 15, 20]]
        tag = "**ours**" if coup else "baseline"
        md.append(f"| {ARCH[mode]} / {tag} | " + " | ".join(cells) + f" | {envelope(rs,.5)} |")
md.append("")

# ---------- Figure 2 + Table 2: encoding sweep -------------------------------
pes = ["nope", "ape", "sin", "rope", "alibi"]
fig, ax = plt.subplots(figsize=(8, 4.3))
w = .38
for i, mode in enumerate(["ar", "diff"]):
    mu = [(stat(sel(prefix="B-pe", mode=mode, pe=p)) or (None, {5: (0, 0, 0)}))[1].get(5, (0, 0, 0))[0] for p in pes]
    sd = [(stat(sel(prefix="B-pe", mode=mode, pe=p)) or (None, {5: (0, 0, 0)}))[1].get(5, (0, 0, 0))[1] for p in pes]
    ax.bar(np.arange(len(pes)) + (i - .5) * w, mu, w, yerr=sd, capsize=3,
           label=ARCH[mode], color=["#7f8c8d", "#e67e22"][i])
ax.set_xticks(range(len(pes))); ax.set_xticklabels([PRETTY[p] for p in pes])
ax.set_ylabel("in-distribution accuracy (%)"); ax.set_ylim(0, 105)
ax.set_title("Positional encodings do not transfer between architectures")
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_encodings.png", dpi=160)

md.append("## Table 2 — Positional encoding sweep (place-value ids)\n")
md.append("| encoding | AR in-dist | AR H*(50%) | Diffusion in-dist | Diffusion H*(50%) |")
md.append("|---|---|---|---|---|")
for p in pes:
    row = [PRETTY[p]]
    for mode in ("ar", "diff"):
        rs = sel(prefix="B-pe", mode=mode, pe=p)
        g = stat(rs)
        row += [f"{g[1][5][0]:.1f}±{g[1][5][1]:.0f}" if g and 5 in g[1] else "–",
                str(envelope(rs, .5)) if rs else "–"]
    md.append("| " + " | ".join(row) + " |")
md.append("")

# ---------- Table 3: component ablation --------------------------------------
md.append("## Table 3 — Component ablation (addition, distance rule)\n")
md.append("| segments | random offset | architecture | 6d | 8d | 12d | H*(50%) |")
md.append("|---|---|---|---|---|---|---|")
for seg in (0, 1):
    for off in (0, 20):
        for mode in ("ar", "diff"):
            rs = sel(prefix="C-abl", segments=seg, max_offset=off, mode=mode)
            g = stat(rs)
            if not g:
                continue
            _, st = g
            cells = [f"{st[d][0]:.1f}" if d in st else "–" for d in (6, 8, 12)]
            md.append(f"| {'yes' if seg else 'no'} | {'yes' if off else 'no'} | "
                      f"{ARCH[mode]} | " + " | ".join(cells) + f" | {envelope(rs,.5)} |")
md.append("")

# ---------- Figure 3: denoising steps (diffusion only) -----------------------
nfe_runs = [r for r in R if r.get("nfe") and r["name"].startswith("A-main")]
if nfe_runs:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    steps = sorted({int(k) for r in nfe_runs for k in r["nfe"]})
    for d, col in [(5, "#1f77b4"), (6, "#2ca02c"), (8, "#d62728")]:
        ys = [np.mean([r["nfe"][str(s)].get(f"d{d}", 0) for r in nfe_runs if str(s) in r["nfe"]]) * 100
              for s in steps]
        ax.plot(steps, ys, marker="o", color=col, label=f"{d} digits")
    ax.set_xscale("log", base=2); ax.set_xlabel("denoising passes (compute at inference)")
    ax.set_ylabel("exact-match accuracy (%)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Diffusion-only knob: accuracy vs number of refinement passes")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_denoising_steps.png", dpi=160)
    md.append("## Figure 3 — accuracy vs denoising passes\n")
    md.append("Autoregressive models have no equivalent knob; this is compute "
              "spent purely at inference.\n")

# ---------- Table 4: multiplication ------------------------------------------
mul = [r for r in R if r["name"].startswith("D-mul")]
if mul:
    md.append("## Table 4 — Multiplication (place value is NOT the algorithm)\n")
    md.append("| configuration | " + " | ".join(f"{d}d" for d in [3, 4, 5, 6, 7]) + " |")
    md.append("|---" * 6 + "|")
    for mode in ("ar", "diff"):
        for coup in (0, 1):
            g = stat(sel(prefix="D-mul", coupled=coup, mode=mode))
            if not g:
                continue
            _, st = g
            cells = [f"{st[d][0]:.1f}±{st[d][1]:.0f}" if d in st else "–" for d in [3, 4, 5, 6, 7]]
            tag = "**ours**" if coup else "baseline"
            md.append(f"| {ARCH[mode]} / {tag} | " + " | ".join(cells) + " |")
    md.append("")

# ---------- partial credit ----------------------------------------------------
dg = [r for r in R if r.get("digit_acc") and r["name"].startswith("A-main")]
if dg:
    md.append("## Table 5 — Per-digit accuracy (partial credit)\n")
    md.append("| configuration | 6d | 8d | 12d | 20d |")
    md.append("|---|---|---|---|---|")
    for mode in ("ar", "diff"):
        for coup in (0, 1):
            rs = [r for r in dg if r["args"]["mode"] == mode and r["args"]["coupled"] == coup
                  and r["args"]["pe"] == "alibi"]
            if not rs:
                continue
            g = stat(rs, key="digit_acc")
            _, st = g
            cells = [f"{st[d][0]:.1f}" if d in st else "–" for d in (6, 8, 12, 20)]
            tag = "**ours**" if coup else "baseline"
            md.append(f"| {ARCH[mode]} / {tag} | " + " | ".join(cells) + " |")
    md.append("")

open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md[:40]))
print(f"\nwrote figures + summary to {OUT}")
