"""Addition with optional significance-aligned position ids (the method).

Baseline (--coupled 0): tokens numbered by sequence index, the standard scheme.
Method   (--coupled 1): tokens numbered by PLACE VALUE, so digits that must be
combined share an id, plus a random per-example offset so the model keys on
relative place value and sees large ids during training.

Units-first output order, which is what makes the place-value id of answer slot
j simply j+1 -- known before generation starts, for both architectures.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import build_op_dataset, build_seq_dataset, Tokenizer
from model import Transformer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ar", "diff"], required=True)
    p.add_argument("--pe", default="ape", choices=["nope", "ape", "sin", "rope", "alibi"])
    p.add_argument("--coupled", type=int, default=1)
    p.add_argument("--op", choices=["add", "mul", "sub", "parity", "reverse"], default="add")
    p.add_argument("--randpos", type=int, default=0,
                   help="randomized positional encodings (Ruoss et al. 2023): sample a sorted\n                         random subset of a much larger index range, so large indices are seen\n                         in training. A published length-generalization baseline.")
    p.add_argument("--segments", type=int, default=1)
    p.add_argument("--abacus", type=int, default=0,
                   help="Abacus-style (McLeish et al. 2024): significance as a learned\n                         embedding added to the token embedding, rather than as position ids")
    p.add_argument("--max_offset", type=int, default=20)
    p.add_argument("--digits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=200000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--fixed_t", type=float, default=0.0,
                   help="train the denoiser at a single noise level instead of sampling t. "
                        "fixed_t=1.0 means every example is fully masked, i.e. one-shot "
                        "bidirectional prediction with NO iterative refinement - which "
                        "separates 'bidirectional attention' from 'multiple passes' as "
                        "explanations for the diffusion advantage.")
    p.add_argument("--ladder", choices=["short", "long"], default="short",
                   help="long = train <=20 digits, test to 100, matching the published "
                        "arithmetic length-generalization protocol")
    p.add_argument("--base", type=int, default=10,
                   help="numeric base; tests whether the effect is about place value in "
                        "general or about decimal digits specifically")
    p.add_argument("--per_instance", type=int, default=0,
                   help="save per-instance correctness so confidence intervals can be "
                        "bootstrapped over test items, not only over seeds")
    p.add_argument("--nfe_sweep", type=int, default=0,
                   help="accuracy-vs-denoising-passes sweep; 7x the eval cost, so off by default")
    p.add_argument("--out", default="runs/dev")
    return p.parse_args()


# Ladder and canvas depend on the operation: multiplication answers are twice
# as long as their operands, so it uses a shorter ladder to keep cost sane.
# "long" matches the protocol used by the published arithmetic work (Abacus
# trains on <=20 digits and reports generalization to 120), so our numbers are
# comparable to theirs rather than only to our own baseline.
LADDER_LONG = {"add": [20, 25, 30, 40, 50, 60, 80, 100],
               "sub": [20, 25, 30, 40, 50, 60, 80, 100],
               "mul": [5, 6, 7, 8, 10],
               "parity": [20, 30, 40, 60, 80, 100],
               "reverse": [20, 30, 40, 60, 80, 100]}

LADDER = {"add": [5, 6, 7, 8, 10, 12, 15, 20],
          "sub": [5, 6, 7, 8, 10, 12, 15, 20],
          "mul": [3, 4, 5, 6, 7],
          "parity": [10, 12, 15, 20, 30, 40],
          "reverse": [10, 12, 15, 20, 30, 40]}
TEST_DIGITS, MAX_TEST, MAX_PROMPT, CANVAS = None, None, None, None


def set_sizes(op, ladder="short"):
    global TEST_DIGITS, MAX_TEST, MAX_PROMPT, CANVAS
    TEST_DIGITS = (LADDER_LONG if ladder == "long" else LADDER)[op]
    MAX_TEST = max(TEST_DIGITS)
    if op in ("parity", "reverse"):
        MAX_PROMPT = MAX_TEST + 2
        CANVAS = MAX_TEST + 3
    else:
        MAX_PROMPT = 2 * MAX_TEST + 2
        CANVAS = (2 * MAX_TEST + 3) if op == "mul" else (MAX_TEST + 3)


def load(n, digits, seed, exact, args):
    if args.op in ("parity", "reverse"):
        return tuple(torch.from_numpy(x) if hasattr(x, "shape") else x
                     for x in build_seq_dataset(
                         n, digits, MAX_PROMPT, CANVAS, seed=seed, exact=exact,
                         max_offset=args.max_offset if args.coupled else 0,
                         task=args.op))
    P, T, pm, PP, TP, PS, TS, tok = build_op_dataset(
        n, digits, MAX_PROMPT, CANVAS, seed=seed, exact=exact, op=args.op,
        reverse=True, max_offset=args.max_offset if args.coupled else 0,
        base=args.base)
    return (torch.from_numpy(P), torch.from_numpy(T), torch.from_numpy(pm),
            torch.from_numpy(PP), torch.from_numpy(TP),
            torch.from_numpy(PS), torch.from_numpy(TS), tok)


def pos_for(PPb, TPb, args, device):
    """Full-sequence position ids, or None to fall back to sequence index."""
    if args.abacus:
        return None          # Abacus keeps sequence-index positions
    if args.randpos:
        B = PPb.shape[0]
        L = PPb.shape[1] + TPb.shape[1]
        hi = 4 * L
        # Seeded per call so a whole generation shares one assignment; training
        # still varies it across batches because the seed follows the data.
        g = torch.Generator(device="cpu").manual_seed(int(PPb.sum().item()) % (2**31))
        idx = torch.stack([torch.randperm(hi, generator=g)[:L].sort().values
                           for _ in range(B)]).to(device)
        return idx
    if not args.coupled:
        return None
    return torch.cat([PPb, TPb], 1).to(device)


def abacus_for(PPb, TPb, args, device):
    """Significance ids for the Abacus variant. Same information as the
    place-value ids, delivered as a token embedding instead of a position."""
    if not args.abacus:
        return None
    return torch.cat([PPb, TPb], 1).to(device)


def seg_for(PSb, TSb, args, device):
    """Segment ids mark operand-A / operand-B / answer. Only meaningful when
    place-value ids are in use, since those deliberately collide across the
    three segments."""
    if not args.coupled or not args.segments:
        return None
    return torch.cat([PSb, TSb], 1).to(device)


def pad_mask_of(pmb, canvas, device):
    return torch.cat([pmb.to(device), torch.ones(pmb.shape[0], canvas, dtype=torch.bool, device=device)], 1)


def diffusion_loss(model, P, T, pm, PP, TP, PS, TS, tok, args, device):
    B, canvas = T.shape
    t_min = 1.0 / canvas
    if args.fixed_t > 0:
        t = torch.full((B,), args.fixed_t, device=device)
    else:
        t = t_min + (1.0 - t_min) * torch.rand(B, device=device)
    noise = torch.rand(B, canvas, device=device) < t[:, None]
    noise[torch.arange(B, device=device), torch.randint(0, canvas, (B,), device=device)] = True
    x_t = torch.where(noise, torch.full_like(T, tok.mask), T)
    logits = model(torch.cat([P, x_t], 1), pad_mask_of(pm, canvas, device),
                   pos_for(PP, TP, args, device),
                   seg_for(PS, TS, args, device),
                   abacus_for(PP, TP, args, device))[:, P.shape[1]:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), T.reshape(-1), reduction="none")
    ce = (ce.view(B, canvas) * noise).sum(1)
    return ((1.0 / t) * ce / canvas).mean()


def ar_loss(model, P, T, pm, PP, TP, PS, TS, tok, args, device):
    canvas = T.shape[1]
    logits = model(torch.cat([P, T], 1), pad_mask_of(pm, canvas, device),
                   pos_for(PP, TP, args, device),
                   seg_for(PS, TS, args, device),
                   abacus_for(PP, TP, args, device))
    pred = logits[:, P.shape[1] - 1: -1]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), T.reshape(-1))


@torch.no_grad()
def sample_diffusion(model, P, pm, PP, TP, PS, TS, tok, args, device):
    B, canvas = P.shape[0], CANVAS
    x = torch.full((B, canvas), tok.mask, device=device, dtype=torch.long)
    pmask = pad_mask_of(pm, canvas, device)
    pos = pos_for(PP, TP, args, device)
    seg = seg_for(PS, TS, args, device)
    aba = abacus_for(PP, TP, args, device)
    pred = None
    for s in range(args.T, 0, -1):
        masked = x == tok.mask
        if not masked.any():
            break
        logits = model(torch.cat([P, x], 1), pmask, pos, seg, aba)[:, P.shape[1]:]
        conf, pred = logits.softmax(-1).max(-1)
        conf = conf.masked_fill(~masked, -1.0)
        remaining = int(canvas * (s - 1) / args.T)
        for b in range(B):
            k = max(0, int(masked[b].sum()) - remaining)
            if k:
                x[b, conf[b].topk(k).indices] = pred[b, conf[b].topk(k).indices]
    still = x == tok.mask
    if still.any() and pred is not None:
        x[still] = pred[still]
    return x


@torch.no_grad()
def sample_ar(model, P, pm, PP, TP, PS, TS, tok, args, device):
    """Greedy left-to-right decoding.

    The auxiliary signals are built ONCE for the full sequence and then sliced,
    which matters for two reasons: randomized positions must stay fixed across
    decode steps (re-drawing them each step showed the model a different
    position assignment every token), and the position rule must match whatever
    training used (pos_for returns None under --abacus, so generation must too).
    """
    B = P.shape[0]
    out = torch.full((B, CANVAS), tok.pad, device=device, dtype=torch.long)
    full_pos = pos_for(PP, TP, args, device)      # None, or (B, MAX_PROMPT+CANVAS)
    full_seg = seg_for(PS, TS, args, device)
    full_aba = abacus_for(PP, TP, args, device)
    cur = P
    curmask = pm.to(device)
    for j in range(CANVAS):
        L = P.shape[1] + j
        pmk = torch.cat([curmask, torch.ones(B, j, dtype=torch.bool, device=device)], 1) if j else curmask
        logits = model(cur, pmk,
                       None if full_pos is None else full_pos[:, :L],
                       None if full_seg is None else full_seg[:, :L],
                       None if full_aba is None else full_aba[:, :L])
        nxt = logits[:, -1].argmax(-1)
        out[:, j] = nxt
        cur = torch.cat([cur, nxt[:, None]], 1)
    return out


def exact_match(pred, gold, tok):
    """Compare answer digits only: drop padding, then truncate at the first EOS.
    Trailing padding is not part of the answer and must not count against it."""
    def norm(seq):
        seq = [t for t in seq if t != tok.pad]
        return seq[: seq.index(tok.eos)] if tok.eos in seq else seq
    return sum(norm(p) == norm(g) for p, g in zip(pred.tolist(), gold.tolist())) / len(gold)


def per_position_acc(pred, gold, tok):
    """Accuracy at each answer position, units-first. Shows *where* long answers
    break: leading digits, or everything past the trained length."""
    def norm(seq):
        seq = [x for x in seq if x != tok.pad]
        return seq[: seq.index(tok.eos)] if tok.eos in seq else seq
    num, den = {}, {}
    for p, g in zip(pred.tolist(), gold.tolist()):
        pn, gn = norm(p), norm(g)
        for i in range(len(gn)):
            den[i] = den.get(i, 0) + 1
            if i < len(pn) and pn[i] == gn[i]:
                num[i] = num.get(i, 0) + 1
    return {i: num.get(i, 0) / den[i] for i in sorted(den)}


def correct_flags(pred, gold, tok):
    def norm(seq):
        seq = [x for x in seq if x != tok.pad]
        return seq[: seq.index(tok.eos)] if tok.eos in seq else seq
    return [int(norm(p) == norm(g)) for p, g in zip(pred.tolist(), gold.tolist())]


def digit_acc(pred, gold, tok):
    """Fraction of answer digits correct (partial credit), aligned from the
    units end. Exact match is the headline metric but hides how close a model
    is; this shows graded degradation."""
    def norm(seq):
        seq = [t for t in seq if t != tok.pad]
        return seq[: seq.index(tok.eos)] if tok.eos in seq else seq
    num = den = 0
    for p, g in zip(pred.tolist(), gold.tolist()):
        pn, gn = norm(p), norm(g)
        den += len(gn)
        num += sum(1 for i in range(len(gn)) if i < len(pn) and pn[i] == gn[i])
    return num / max(den, 1)


@torch.no_grad()
def evaluate(model, args, tok, device):
    model.eval()
    res, dres, pres, flags = {}, {}, {}, {}
    for dd in TEST_DIGITS:
        P, T, pm, PP, TP, PS, TS, _ = load(args.n_eval, dd, 40_000 + dd, True, args)
        accs, dgs, poss, fl = [], [], [], []
        for i in range(0, len(P), args.eval_bs):
            sl = slice(i, i + args.eval_bs)
            ac = dict(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else dict(device_type="cpu", enabled=False)
            with torch.autocast(**ac):
                f = sample_diffusion if args.mode == "diff" else sample_ar
                pred = f(model, P[sl].to(device), pm[sl], PP[sl], TP[sl], PS[sl], TS[sl], tok, args, device)
            gold = T[sl].to(device)
            accs.append(exact_match(pred, gold, tok))
            dgs.append(digit_acc(pred, gold, tok))
            poss.append(per_position_acc(pred, gold, tok))
            if args.per_instance:
                fl += correct_flags(pred, gold, tok)
        res[f"d{dd}"] = float(np.mean(accs))
        dres[f"d{dd}"] = float(np.mean(dgs))
        keys = sorted({k for d in poss for k in d})
        pres[f"d{dd}"] = {str(k): float(np.mean([d[k] for d in poss if k in d])) for k in keys}
        if args.per_instance:
            flags[f"d{dd}"] = fl
    model.train()
    return res, dres, pres, flags


def main():
    args = get_args()
    set_sizes(args.op, args.ladder)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    P, T, pm, PP, TP, PS, TS, tok = load(args.n_train, args.digits, args.seed, False, args)
    max_len = 4 * (MAX_PROMPT + CANVAS) + 8 if args.randpos else \
              MAX_PROMPT + CANVAS + args.max_offset + MAX_TEST + 8
    model = Transformer(len(tok), args.d, args.layers, args.heads, args.pe,
                        causal=(args.mode == "ar"), max_len=max_len).to(device)
    print(f"[{args.mode}/{args.pe}/coupled={args.coupled}] params={model.n_params()/1e6:.2f}M "
          f"train_digits<={args.digits} test={TEST_DIGITS}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else dict(device_type="cpu", enabled=False)
    lossfn = diffusion_loss if args.mode == "diff" else ar_loss

    log, t0 = [], time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            i = torch.randint(0, len(P), (args.bs,))
            with torch.autocast(**amp):
                loss = lossfn(model, P[i].to(device), T[i].to(device), pm[i],
                              PP[i], TP[i], PS[i], TS[i], tok, args, device) / args.accum
            loss.backward(); tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            print(f"step {step:6d} loss {tot:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0:
            e = evaluate(model, args, tok, device)[0]
            print(f"  EVAL {step+1}: {e}", flush=True)

    final, dfinal, pfinal, iflags = evaluate(model, args, tok, device)

    # accuracy vs number of denoising passes (no autoregressive analogue)
    nfe = {}
    if args.mode == "diff" and args.nfe_sweep:
        keep = args.T
        for t_steps in (1, 2, 4, 8, 16, 32):
            args.T = t_steps
            e = evaluate(model, args, tok, device)[0]
            nfe[t_steps] = e
        args.T = keep

    n_par = model.n_params()
    n_emb = model.emb.weight.numel()
    tokens = args.steps * args.bs * args.accum * (MAX_PROMPT + CANVAS)
    compute = {"params": n_par, "non_embedding_params": n_par - n_emb,
               "train_tokens": tokens,
               "train_flops_approx": 6 * (n_par - n_emb) * tokens,
               "inference_passes_per_example": (args.T if args.mode == "diff" else CANVAS),
               "minutes": (time.time() - t0) / 60}
    json.dump({"args": vars(args), "final": final, "digit_acc": dfinal,
               "per_position": pfinal, "instance_flags": iflags,
               "nfe": nfe, "compute": compute, "minutes": (time.time() - t0) / 60},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
