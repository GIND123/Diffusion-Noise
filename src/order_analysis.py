"""Which positions does the diffusion model resolve first?

The headline claim is that masked diffusion benefits more from place-value
alignment because it can choose its decoding order. That is a claim about
mechanism, and so far it has been asserted rather than shown. This records the
pass at which each answer slot is revealed and checks whether the model
discovers the carry order (low-significance digits first) on its own.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Tokenizer
from model import Transformer
import train_add as ta

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--pe", default="alibi")
p.add_argument("--coupled", type=int, default=1)
p.add_argument("--segments", type=int, default=1)
p.add_argument("--abacus", type=int, default=0)
p.add_argument("--randpos", type=int, default=0)
p.add_argument("--max_offset", type=int, default=20)
p.add_argument("--op", default="add")
p.add_argument("--digits", type=int, default=5)
p.add_argument("--n", type=int, default=400)
p.add_argument("--T", type=int, default=16)
p.add_argument("--mode", default="diff")
p.add_argument("--eval_bs", type=int, default=100)
p.add_argument("--out", default="figures")
a = p.parse_args()

ta.set_sizes(a.op)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer(10)
P, T, pm, PP, TP, PS, TS, _ = ta.load(a.n, a.digits, 40_000 + a.digits, True, a)

m = Transformer(len(tok), 384, 6, 6, a.pe, causal=False,
                max_len=ta.MAX_PROMPT + ta.CANVAS + a.max_offset + ta.MAX_TEST + 8).to(dev)
# strict=False: checkpoints saved before the Abacus embedding existed
m.load_state_dict(torch.load(a.ckpt, map_location=dev), strict=False)
m.eval()

reveal = np.full((a.n, ta.CANVAS), np.nan)   # pass index at which each slot was set

with torch.no_grad():
    for lo in range(0, a.n, a.eval_bs):
        sl = slice(lo, lo + a.eval_bs)
        Pb = P[sl].to(dev)
        B = Pb.shape[0]
        x = torch.full((B, ta.CANVAS), tok.mask, device=dev, dtype=torch.long)
        pmask = ta.pad_mask_of(pm[sl], ta.CANVAS, dev)
        pos = ta.pos_for(PP[sl], TP[sl], a, dev)
        seg = ta.seg_for(PS[sl], TS[sl], a, dev)
        aba = ta.abacus_for(PP[sl], TP[sl], a, dev)
        for s in range(a.T, 0, -1):
            masked = x == tok.mask
            if not masked.any():
                break
            logits = m(torch.cat([Pb, x], 1), pmask, pos, seg, aba)[:, Pb.shape[1]:]
            conf, pred = logits.softmax(-1).max(-1)
            conf = conf.masked_fill(~masked, -1.0)
            remaining = int(ta.CANVAS * (s - 1) / a.T)
            step = a.T - s
            for b in range(B):
                k = max(0, int(masked[b].sum()) - remaining)
                if k:
                    idx = conf[b].topk(k).indices
                    x[b, idx] = pred[b, idx]
                    for i in idx.tolist():
                        if np.isnan(reveal[lo + b, i]):
                            reveal[lo + b, i] = step

# answer slots only (units-first, so slot j is significance j+1)
ans_len = a.digits + 1
mean_pass = np.nanmean(reveal[:, :ans_len], axis=0)
sig = np.arange(1, ans_len + 1)
rho = np.corrcoef(sig, mean_pass)[0, 1]

os.makedirs(a.out, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sig, mean_pass, marker="o", color="#1f77b4")
ax.set_xlabel("digit significance (1 = units)")
ax.set_ylabel("mean denoising pass at which the digit is fixed")
ax.set_title(f"Decoding order discovered by the diffusion model\n"
             f"correlation with significance = {rho:+.2f}")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(f"{a.out}/fig4_decoding_order.png", dpi=160)

out = {"mean_pass_by_significance": mean_pass.tolist(),
       "correlation_significance_vs_pass": float(rho),
       "reads": "positive correlation = low-order digits resolved first, i.e. the "
                "model follows the carry chain rather than writing left to right"}
json.dump(out, open(f"{a.out}/decoding_order.json", "w"), indent=2)
print(json.dumps(out, indent=2))
