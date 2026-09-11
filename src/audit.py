"""Numerical audit. Verifies the things that would silently invalidate results."""
import numpy as np, torch, json, glob, os
from data import build_op_dataset, Tokenizer
import train_add as ta

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print("\n=== 1. Are the arithmetic problems actually correct? ===")
for op in ("add", "mul"):
    ta.set_sizes(op)
    tok = Tokenizer(10)
    d = 5 if op == "add" else 3
    P, T, pm, PP, TP, PS, TS, _ = build_op_dataset(
        400, d, ta.MAX_PROMPT, ta.CANVAS, seed=123, exact=True, op=op, max_offset=20)
    bad = 0
    for i in range(len(P)):
        toks = [tok.itos[t] for t in P[i] if t != tok.pad]
        sep = "+" if op == "add" else "|"
        a = int("".join(toks[:toks.index(sep)]))
        b = int("".join(toks[toks.index(sep) + 1:toks.index("=")]))
        ans = [tok.itos[t] for t in T[i] if t != tok.pad]
        ans = ans[: ans.index("[EOS]")]
        got = int("".join(reversed(ans)))          # stored units-first
        want = a + b if op == "add" else a * b
        bad += (got != want)
    check(f"{op}: 400 problems have correct answers", bad == 0, f"{bad} wrong")


print("\n=== 2. Does the metric score correctly? ===")
tok = Tokenizer(10)
ta.set_sizes("add")
g = torch.tensor([[tok.stoi["3"], tok.stoi["2"], tok.eos, tok.pad, tok.pad]])
cases = [
    ("identical", [tok.stoi["3"], tok.stoi["2"], tok.eos, tok.pad, tok.pad], 1.0),
    ("EOS late + padding between", [tok.stoi["3"], tok.stoi["2"], tok.pad, tok.eos, tok.pad], 1.0),
    ("one digit wrong", [tok.stoi["3"], tok.stoi["9"], tok.eos, tok.pad, tok.pad], 0.0),
    ("extra digit", [tok.stoi["3"], tok.stoi["2"], tok.stoi["1"], tok.eos, tok.pad], 0.0),
    ("missing digit", [tok.stoi["3"], tok.eos, tok.pad, tok.pad, tok.pad], 0.0),
]
for nm, p, want in cases:
    got = ta.exact_match(torch.tensor([p]), g, tok)
    check(f"metric: {nm} -> {want}", got == want, f"got {got}")


print("\n=== 3. Do place-value ids leak the answer length? ===")
ta.set_sizes("add")
P, T, pm, PP, TP, PS, TS, _ = build_op_dataset(
    200, 5, ta.MAX_PROMPT, ta.CANVAS, seed=7, exact=True, op="add", max_offset=20)
# answer length vs number of non-zero target position ids
lens = [(T[i] != tok.pad).sum() for i in range(len(T))]
nz = [(TP[i] != 0).sum() for i in range(len(TP))]
if len({int(x) for x in nz}) == 1:
    check("target position ids do NOT reveal answer length", True,
          "id count is constant, so it carries no length information")
else:
    leak = np.corrcoef(lens, nz)[0, 1]
    check("target position ids do NOT reveal answer length", abs(leak) < 0.01,
          f"correlation={leak:.3f} (1.0 means the model is told the answer length)")

varying = len({int(x) for x in nz})
check("answer-length signal is constant across examples", varying == 1,
      f"{varying} distinct non-zero counts")


print("\n=== 4. Train/test overlap ===")
ta.set_sizes("add")
tr = build_op_dataset(20000, 5, ta.MAX_PROMPT, ta.CANVAS, seed=0, exact=False, op="add", max_offset=20)
te = build_op_dataset(500, 5, ta.MAX_PROMPT, ta.CANVAS, seed=40_005, exact=True, op="add", max_offset=20)
def keys(D):
    out = set()
    for i in range(len(D[0])):
        out.add(tuple(int(t) for t in D[0][i] if t != tok.pad))
    return out
ov = keys(tr) & keys(te)
check("no overlap between train and 5-digit test problems", len(ov) == 0, f"{len(ov)} shared")


print("\n=== 5. Are reported means reproducible from the raw run files? ===")
runs = os.environ.get("RUNS", "runs")
by = {}
for f in glob.glob(f"{runs}/A-main-*/result.json"):
    r = json.load(open(f))
    a = r["args"]
    if a.get("pe") != "alibi":
        continue
    by.setdefault((a["mode"], a["coupled"]), []).append(r["final"].get("d6", 0))
for k, v in sorted(by.items()):
    if len(v) >= 2:
        print(f"    {k[0]} coupled={k[1]}: n={len(v)} d6 values={[round(x*100,1) for x in v]} "
              f"mean={np.mean(v)*100:.1f} sd={np.std(v)*100:.1f}")
check("A-main groups have 3 seeds each", all(len(v) == 3 for v in by.values()) if by else False,
      f"group sizes {[len(v) for v in by.values()]}")


print("\n=== 6. Accuracy values are in range ===")
bad = []
for f in glob.glob(f"{runs}/*/result.json"):
    r = json.load(open(f))
    for k, v in r.get("final", {}).items():
        if not (0.0 <= v <= 1.0):
            bad.append((os.path.basename(os.path.dirname(f)), k, v))
check("all accuracies within [0,1]", not bad, str(bad[:3]))

print(f"\n=== AUDIT: {len(fails)} failure(s) ===")
for f in fails:
    print("  !", f)
