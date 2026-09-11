"""Train AR or masked-diffusion on star-graph; evaluate in-distribution and OOD.

OOD uses two independent dials:
  path_len  -> longer target (length generalization; tests the fixed canvas, H1)
  deg       -> harder search, target length UNCHANGED (planning difficulty)
Separating them is what distinguishes "longer" from "harder".
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import build_dataset, build_sort_dataset, build_add_dataset, Tokenizer
from model import Transformer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ar", "diff"], required=True)
    p.add_argument("--task", choices=["star", "sort", "add"], default="star")
    p.add_argument("--digits", type=int, default=5, help="add: max train operand digits")
    p.add_argument("--order", choices=["lsd", "msd"], default="lsd",
                   help="add: output digit order. lsd = carry flows left-to-right (AR-friendly); "
                        "msd = leading digit needs the whole carry chain first (AR-hostile)")
    p.add_argument("--k", type=int, default=8, help="sort: list length")
    p.add_argument("--pe", default="ape", choices=["nope", "ape", "sin", "rope", "alibi"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_nodes", type=int, default=50)
    p.add_argument("--train_deg", type=int, default=2)
    p.add_argument("--train_path", type=int, default=5)
    p.add_argument("--n_train", type=int, default=100000)
    p.add_argument("--n_eval", type=int, default=500)
    p.add_argument("--max_prompt", type=int, default=0, help="0 = auto")
    p.add_argument("--canvas", type=int, default=0, help="0 = auto")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--bs", type=int, default=64, help="micro-batch; effective batch = bs*accum")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--eval_bs", type=int, default=100)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--T", type=int, default=24, help="diffusion sampling steps")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--out", default="runs/dev")
    return p.parse_args()


def ood_splits(train_deg, train_path):
    """Two independent dials. Length changes the target size (tests the fixed
    canvas); degree changes search difficulty at constant target size."""
    s = [(train_deg, train_path, "id")]
    for p in [train_path + 1, train_path + 2, train_path + 4, train_path + 7]:
        s.append((train_deg, p, f"len{p}"))
    for d in [train_deg + 1, train_deg + 3]:
        s.append((d, train_path, f"deg{d}"))
    return s


def auto_sizes(train_deg, train_path):
    sp = ood_splits(train_deg, train_path)
    max_prompt = max(d * p for d, p, _ in sp) * 4 + 4
    canvas = max(2 * (p + 1) for _, p, _ in sp)
    return max_prompt, canvas


def sort_splits(k):
    return [(k, "id")] + [(kk, f"len{kk}") for kk in (k + 4, k + 8, 2 * k + 8)]


def add_splits(d):
    """Train on 1..d digits; test on strictly longer operands."""
    return [(d, "id")] + [(dd, f"d{dd}") for dd in (d + 1, d + 2, d + 3, d + 5, d + 7)]


def add_sizes(d):
    md = max(dd for dd, _ in add_splits(d))
    return 2 * md + 2, md + 3


def sort_sizes(k):
    mk = max(kk for kk, _ in sort_splits(k))
    return 2 * mk + 2, 2 * mk


def make_pad_mask(P, pmask, canvas, device):
    B = P.shape[0]
    tgt = torch.ones(B, canvas, dtype=torch.bool, device=device)
    return torch.cat([pmask.to(device), tgt], 1)


def diffusion_loss(model, P, T, pmask, tok, device):
    B, canvas = T.shape
    # Clamp t so E[#masked] >= 1. Without this the 1/t weight explodes (~40x)
    # on the rare near-zero-t draws and swamps the gradient.
    t_min = 1.0 / canvas
    t = t_min + (1.0 - t_min) * torch.rand(B, device=device)
    noise = torch.rand(B, canvas, device=device) < t[:, None]
    # guarantee at least one masked position per example
    forced = torch.randint(0, canvas, (B,), device=device)
    noise[torch.arange(B, device=device), forced] = True

    x_t = torch.where(noise, torch.full_like(T, tok.mask), T)
    seq = torch.cat([P, x_t], 1)
    logits = model(seq, make_pad_mask(P, pmask, canvas, device))[:, P.shape[1]:]

    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), T.reshape(-1), reduction="none")
    ce = (ce.view(B, canvas) * noise).sum(1)
    return ((1.0 / t) * ce / canvas).mean()


def ar_loss(model, P, T, pmask, tok, device):
    canvas = T.shape[1]
    seq = torch.cat([P, T], 1)
    logits = model(seq, make_pad_mask(P, pmask, canvas, device))
    pred = logits[:, P.shape[1] - 1: -1]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), T.reshape(-1))


@torch.no_grad()
def sample_diffusion(model, P, pmask, tok, canvas, T_steps, device):
    B = P.shape[0]
    x = torch.full((B, canvas), tok.mask, device=device, dtype=torch.long)
    pm = make_pad_mask(P, pmask, canvas, device)
    for s in range(T_steps, 0, -1):
        masked = x == tok.mask
        if not masked.any():
            break
        logits = model(torch.cat([P, x], 1), pm)[:, P.shape[1]:]
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)
        conf = conf.masked_fill(~masked, -1.0)  # only consider masked slots

        target_remaining = int(canvas * (s - 1) / T_steps)
        for b in range(B):
            nm = int(masked[b].sum())
            k = max(0, nm - target_remaining)
            if k == 0:
                continue
            idx = conf[b].topk(k).indices
            x[b, idx] = pred[b, idx]
    still = x == tok.mask
    if still.any():
        x[still] = pred[still]
    return x


@torch.no_grad()
def sample_ar(model, P, pmask, tok, canvas, device):
    B = P.shape[0]
    out = torch.full((B, canvas), tok.pad, device=device, dtype=torch.long)
    cur = P
    curmask = pmask.to(device)
    for j in range(canvas):
        pm = torch.cat([curmask, torch.ones(B, j, dtype=torch.bool, device=device)], 1) if j else curmask
        logits = model(cur, pm)
        nxt = logits[:, -1].argmax(-1)
        out[:, j] = nxt
        cur = torch.cat([cur, nxt[:, None]], 1)
    return out


def exact_match(pred, gold, tok):
    """Compare up to and including the first EOS; PAD after that is ignored."""
    ok = 0
    for p, g in zip(pred.tolist(), gold.tolist()):
        ge = g.index(tok.eos) if tok.eos in g else len(g)
        pe = p.index(tok.eos) if tok.eos in p else len(p)
        if p[:pe] == g[:ge]:
            ok += 1
    return ok / len(gold)


@torch.no_grad()
def evaluate(model, args, tok, device):
    model.eval()
    res = {}
    specs = (sort_splits(args.k) if args.task == "sort"
             else add_splits(args.digits) if args.task == "add"
             else ood_splits(args.train_deg, args.train_path))
    for spec in specs:
        if args.task == "add":
            dd, label = spec
            P, T, pmask, _ = build_add_dataset(args.n_eval, dd, args.max_prompt, args.canvas,
                                               seed=30_000 + dd, exact=True,
                                               reverse=(args.order == "lsd"))
        elif args.task == "sort":
            kk, label = spec
            P, T, pmask, _ = build_sort_dataset(args.n_eval, args.num_nodes, kk,
                                                args.max_prompt, args.canvas, seed=20_000 + kk)
        else:
            deg, plen, label = spec
            P, T, pmask, _ = build_dataset(args.n_eval, args.num_nodes, deg, plen,
                                           args.max_prompt, args.canvas, seed=10_000 + deg * 100 + plen)
        P = torch.from_numpy(P).to(device)
        T_ = torch.from_numpy(T).to(device)
        pm = torch.from_numpy(pmask).to(device)
        accs = []
        for i in range(0, len(P), args.eval_bs):
            sl = slice(i, i + args.eval_bs)
            ac = dict(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else dict(device_type="cpu", enabled=False)
            with torch.autocast(**ac):
                if args.mode == "diff":
                    pred = sample_diffusion(model, P[sl], pm[sl], tok, args.canvas, args.T, device)
                else:
                    pred = sample_ar(model, P[sl], pm[sl], tok, args.canvas, device)
            accs.append(exact_match(pred, T_[sl], tok))
        res[label] = float(np.mean(accs))
    model.train()
    return res


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ap, ac = (sort_sizes(args.k) if args.task == "sort"
              else add_sizes(args.digits) if args.task == "add"
              else auto_sizes(args.train_deg, args.train_path))
    if not args.max_prompt:
        args.max_prompt = ap
    if not args.canvas:
        args.canvas = ac
    _lbl = ([l for _, l in sort_splits(args.k)] if args.task == "sort"
            else [l for _, l in add_splits(args.digits)] if args.task == "add"
            else [l for _, _, l in ood_splits(args.train_deg, args.train_path)])
    print(f"task={args.task} max_prompt={args.max_prompt} canvas={args.canvas} splits={_lbl}", flush=True)

    tok = Tokenizer(args.num_nodes)
    if args.task == "add":
        P, T, pmask, _ = build_add_dataset(args.n_train, args.digits, args.max_prompt,
                                           args.canvas, seed=args.seed,
                                           reverse=(args.order == "lsd"))
    elif args.task == "sort":
        P, T, pmask, _ = build_sort_dataset(args.n_train, args.num_nodes, args.k,
                                            args.max_prompt, args.canvas, seed=args.seed)
    else:
        P, T, pmask, _ = build_dataset(args.n_train, args.num_nodes, args.train_deg, args.train_path,
                                       args.max_prompt, args.canvas, seed=args.seed)
    P = torch.from_numpy(P)
    T = torch.from_numpy(T)
    pmask = torch.from_numpy(pmask)

    model = Transformer(len(tok), args.d, args.layers, args.heads, args.pe,
                        causal=(args.mode == "ar"), max_len=args.max_prompt + args.canvas).to(device)
    print(f"[{args.mode}/{args.pe}] params={model.n_params()/1e6:.2f}M device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    log = []
    t0 = time.time()
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else dict(device_type="cpu", enabled=False)
    lossfn = diffusion_loss if args.mode == "diff" else ar_loss
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.accum):
            idx = torch.randint(0, len(P), (args.bs,))
            pb, tb, mb = P[idx].to(device), T[idx].to(device), pmask[idx].to(device)
            with torch.autocast(**amp):
                loss = lossfn(model, pb, tb, mb, tok, device) / args.accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0:
            print(f"step {step:5d} loss {tot:.4f} ({time.time()-t0:.0f}s)", flush=True)
            log.append({"step": step, "loss": tot})
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, args, tok, device)
            print(f"  EVAL step {step+1}: {acc}", flush=True)
            log.append({"step": step + 1, "eval": acc})

    final = evaluate(model, args, tok, device)
    out = {"args": vars(args), "params_M": model.n_params() / 1e6,
           "final": final, "log": log, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
