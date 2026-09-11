"""Dump the denoising trajectory of a trained masked-diffusion model.

Shows generation as it actually happens: every answer position starts hidden and
is revealed over successive refinement passes, in an order the model picks by
confidence. This is the part that has no autoregressive equivalent.
"""
import argparse, torch
from data import build_add_dataset, Tokenizer
from model import Transformer
from train import make_pad_mask, add_sizes

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--pe", default="alibi")
p.add_argument("--digits", type=int, default=5)
p.add_argument("--order", default="lsd")
p.add_argument("--n", type=int, default=3)
p.add_argument("--T", type=int, default=16)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
max_prompt, canvas = add_sizes(a.digits)
tok = Tokenizer(10)

P, T, pm, _ = build_add_dataset(a.n, a.digits, max_prompt, canvas, seed=777,
                                exact=True, reverse=(a.order == "lsd"))
P = torch.from_numpy(P).to(dev); T = torch.from_numpy(T).to(dev); pm = torch.from_numpy(pm).to(dev)

m = Transformer(len(tok), 384, 6, 6, a.pe, causal=False, max_len=max_prompt + canvas).to(dev)
m.load_state_dict(torch.load(a.ckpt, map_location=dev)); m.eval()


def show(row):
    return " ".join("__" if t == tok.mask else ("." if t == tok.pad else tok.itos[t]) for t in row)


x = torch.full((a.n, canvas), tok.mask, device=dev, dtype=torch.long)
pmask = make_pad_mask(P, pm, canvas, dev)

for i in range(a.n):
    q = " ".join(tok.itos[t] for t in P[i].tolist() if t != tok.pad)
    print(f"\n=== problem {i}: {q}   (gold: {show(T[i].tolist())}) ===")
    print(f"  pass  0: {show(x[i].tolist())}")

with torch.no_grad():
    for s in range(a.T, 0, -1):
        masked = x == tok.mask
        if not masked.any():
            break
        logits = m(torch.cat([P, x], 1), pmask)[:, P.shape[1]:]
        conf, pred = logits.softmax(-1).max(-1)
        conf = conf.masked_fill(~masked, -1.0)
        remaining = int(canvas * (s - 1) / a.T)
        for b in range(a.n):
            k = max(0, int(masked[b].sum()) - remaining)
            if k:
                idx = conf[b].topk(k).indices
                x[b, idx] = pred[b, idx]
        step = a.T - s + 1
        for i in range(a.n):
            print(f"  pass {step:2d}: {show(x[i].tolist())}" if i == 0 else
                  f"           {show(x[i].tolist())}")

print("\nlegend: __ = still hidden, . = padding, digits = revealed")
