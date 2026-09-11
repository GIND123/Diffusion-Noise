"""Print model predictions against gold to diagnose eval failures."""
import argparse, torch
from data import build_dataset, Tokenizer
from model import Transformer
from train import sample_diffusion, sample_ar, make_pad_mask

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--mode", default="diff")
p.add_argument("--pe", default="ape")
p.add_argument("--n", type=int, default=6)
p.add_argument("--max_prompt", type=int, default=112)
p.add_argument("--canvas", type=int, default=24)
p.add_argument("--T", type=int, default=24)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer(50)
P, T, pm, _ = build_dataset(a.n, 50, 2, 5, a.max_prompt, a.canvas, seed=10_000 + 2 * 100 + 5)
P, T, pm = torch.from_numpy(P).to(dev), torch.from_numpy(T).to(dev), torch.from_numpy(pm).to(dev)

m = Transformer(len(tok), 384, 6, 6, a.pe, causal=(a.mode == "ar"), max_len=a.max_prompt + a.canvas).to(dev)
m.load_state_dict(torch.load(a.ckpt, map_location=dev))
m.eval()

with torch.no_grad():
    pred = (sample_diffusion(m, P, pm, tok, a.canvas, a.T, dev) if a.mode == "diff"
            else sample_ar(m, P, pm, tok, a.canvas, dev))

for i in range(a.n):
    g = [tok.itos[t] for t in T[i].tolist()]
    q = [tok.itos[t] for t in pred[i].tolist()]
    print(f"\n--- ex {i} ---")
    print("  prompt:", " ".join(tok.itos[t] for t in P[i].tolist() if t != tok.pad)[:150])
    print("  gold  :", " ".join(g))
    print("  pred  :", " ".join(q))

# also: teacher-forced accuracy, isolating the sampler from the model
with torch.no_grad():
    xt = torch.full_like(T, tok.mask)
    logits = m(torch.cat([P, xt], 1), make_pad_mask(P, pm, a.canvas, dev))[:, P.shape[1]:]
    print("\nall-masked one-shot argmax vs gold (per-token acc):",
          (logits.argmax(-1) == T).float().mean().item())
