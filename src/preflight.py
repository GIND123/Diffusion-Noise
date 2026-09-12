"""Pre-flight validation. Every grid must pass this before consuming compute.

Motivated by two bugs that were only caught after full grids had run, both of
the same class: the generation path silently disagreeing with the training path
(randomized positions redrawn per decode step; Abacus using place-value
positions at generation but sequential ones in training). Both produced 0%
in-distribution accuracy and would have been reported as published methods
failing.

Three layers, cheapest first:
  1. data      - problems are arithmetically correct, ids leak nothing
  2. plumbing  - training and generation see identical auxiliary signals
  3. learning  - every configuration can overfit a tiny set (catches silent
                 train/test mismatch that layers 1-2 cannot see)

Exit code is non-zero on any failure so a job script can abort.
"""
import itertools, json, sys, math
import numpy as np
import torch
import torch.nn.functional as F

from data import build_op_dataset, build_seq_dataset, Tokenizer
from model import Transformer
import train_add as ta

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


class Args:
    def __init__(self, **kw):
        d = dict(mode="diff", op="add", pe="alibi", coupled=1, segments=1, abacus=0,
                 randpos=0, fixed_t=0.0, base=10, max_offset=20, digits=5, seed=0,
                 T=8, eval_bs=32, per_instance=0, nfe_sweep=0, k=8,
                 train_deg=2, train_path=3, num_nodes=50, n_eval=32)
        d.update(kw)
        self.__dict__.update(d)


# ---------------------------------------------------------------- layer 1: data
print("\n=== 1. data integrity ===")
for op, base in itertools.product(("add", "sub", "mul"), (10, 2, 16)):
    ta.set_sizes(op)
    tok = Tokenizer(16)
    d = 5 if op != "mul" else 3
    P, T, pm, PP, TP, PS, TS, _ = build_op_dataset(
        200, d, ta.MAX_PROMPT, ta.CANVAS, seed=11, exact=True, op=op,
        max_offset=20, base=base)
    bad = 0
    sym = {"add": "+", "mul": "|", "sub": ","}[op]
    for i in range(len(P)):
        toks = [tok.itos[x] for x in P[i] if x != tok.pad]
        a = int("".join(toks[:toks.index(sym)]), base)
        b = int("".join(toks[toks.index(sym) + 1:toks.index("=")]), base)
        ans = [tok.itos[x] for x in T[i] if x != tok.pad]
        ans = ans[: ans.index("[EOS]")]
        got = int("".join(reversed(ans)), base)
        want = {"add": a + b, "mul": a * b, "sub": a - b}[op]
        bad += got != want
    check(f"{op} base-{base} arithmetic", bad == 0, f"{bad}/200 wrong")

    nz = {int((TP[i] != 0).sum()) for i in range(len(TP))}
    check(f"{op} base-{base} ids leak no length", len(nz) == 1, f"{len(nz)} distinct id counts")

for task in ("parity", "reverse"):
    ta.set_sizes(task)
    tok = Tokenizer(10)
    P, T, pm, PP, TP, PS, TS, _ = build_seq_dataset(
        200, 10, ta.MAX_PROMPT, ta.CANVAS, seed=11, exact=True, max_offset=20, task=task)
    bad = 0
    for i in range(len(P)):
        xs = [int(tok.itos[x]) for x in P[i] if x != tok.pad and tok.itos[x] != "="]
        ys = [int(tok.itos[x]) for x in T[i] if x != tok.pad and tok.itos[x] != "[EOS]"]
        if task == "parity":
            run, exp = 0, []
            for v in xs:
                run ^= v
                exp.append(run)
        else:
            exp = list(reversed(xs))
        bad += ys != exp
    check(f"{task} targets", bad == 0, f"{bad}/200 wrong")

# ------------------------------------------------------------ layer 2: plumbing
print("\n=== 2. training and generation see the same signals ===")
CONFIGS = [
    dict(name="place-value", coupled=1, segments=1),
    dict(name="sequential", coupled=0, segments=0),
    dict(name="abacus", coupled=1, segments=1, abacus=1),
    dict(name="randomized-PE", coupled=0, segments=0, randpos=1),
]
for cfg in CONFIGS:
    nm = cfg.pop("name")
    a = Args(**cfg)
    ta.set_sizes(a.op)
    P, T, pm, PP, TP, PS, TS, tok = ta.load(8, a.digits, 5, True, a)
    dev = "cpu"
    tr_pos = ta.pos_for(PP, TP, a, dev)
    tr_aba = ta.abacus_for(PP, TP, a, dev)
    # regenerate: randomized ids must be reproducible for the same batch,
    # otherwise a multi-step generation sees a different assignment each step
    again = ta.pos_for(PP, TP, a, dev)
    stable = (tr_pos is None and again is None) or torch.equal(tr_pos, again)
    check(f"{nm}: position ids stable across calls", stable,
          "redrawing per call breaks multi-step decoding")
    both_none = (tr_pos is None) == (ta.pos_for(PP, TP, a, dev) is None)
    check(f"{nm}: abacus/position modes consistent",
          not (a.abacus and tr_pos is not None) and (tr_aba is not None) == bool(a.abacus),
          "abacus must keep sequence-index positions")
    cfg["name"] = nm

# ------------------------------------------------------------- layer 3: learning
print("\n=== 3. every configuration can fit a tiny set ===")
print("  (a configuration that cannot overfit 64 examples is broken, and this")
print("   catches train/generation mismatch that layers 1-2 cannot see)")
dev = "cuda" if torch.cuda.is_available() else "cpu"
GRID = [
    ("place-value/diff", dict(mode="diff", coupled=1, segments=1)),
    ("place-value/ar", dict(mode="ar", coupled=1, segments=1)),
    ("sequential/diff", dict(mode="diff", coupled=0, segments=0)),
    ("sequential/ar", dict(mode="ar", coupled=0, segments=0)),
    ("abacus/ar", dict(mode="ar", coupled=1, segments=1, abacus=1)),
    ("abacus/diff", dict(mode="diff", coupled=1, segments=1, abacus=1)),
    ("randpos/ar", dict(mode="ar", coupled=0, segments=0, randpos=1)),
    ("randpos/diff", dict(mode="diff", coupled=0, segments=0, randpos=1)),
    ("one-shot/diff", dict(mode="diff", coupled=1, segments=1, fixed_t=1.0)),
    ("base2/diff", dict(mode="diff", coupled=1, segments=1, base=2)),
    ("parity/diff", dict(mode="diff", op="parity", coupled=1, segments=1, digits=10)),
    ("reverse/ar", dict(mode="ar", op="reverse", coupled=1, segments=1, digits=10)),
]
N, STEPS = 64, 400
for nm, cfg in GRID:
    a = Args(n_eval=N, T=8, **cfg)
    ta.set_sizes(a.op)
    torch.manual_seed(0)
    P, T, pm, PP, TP, PS, TS, tok = ta.load(N, a.digits, 3, True, a)
    max_len = (4 * (ta.MAX_PROMPT + ta.CANVAS) + 8 if a.randpos
               else ta.MAX_PROMPT + ta.CANVAS + a.max_offset + ta.MAX_TEST + 8)
    m = Transformer(len(tok), 128, 3, 4, a.pe, causal=(a.mode == "ar"), max_len=max_len).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    lossfn = ta.diffusion_loss if a.mode == "diff" else ta.ar_loss
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = lossfn(m, P.to(dev), T.to(dev), pm, PP, TP, PS, TS, tok, a, dev)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        f = ta.sample_diffusion if a.mode == "diff" else ta.sample_ar
        pred = f(m, P.to(dev), pm, PP, TP, PS, TS, tok, a, dev)
        acc = ta.exact_match(pred, T.to(dev), tok)
    check(f"{nm} overfits 64 examples", acc >= 0.5, f"train accuracy {acc*100:.0f}%")

print(f"\n=== PRE-FLIGHT: {len(FAIL)} failure(s) ===")
for f in FAIL:
    print("  !", f)
sys.exit(1 if FAIL else 0)
