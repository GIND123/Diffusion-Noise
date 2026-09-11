"""Why does diffusion score 0 with place-value ids? Look at the predictions."""
import argparse, torch
from data import build_add_dataset_coupled, Tokenizer
from model import Transformer
import train_add as ta

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--pe", default="alibi")
p.add_argument("--mode", default="diff")
p.add_argument("--coupled", type=int, default=1)
p.add_argument("--max_offset", type=int, default=20)
p.add_argument("--digits", type=int, default=5)
p.add_argument("--n", type=int, default=4)
p.add_argument("--T", type=int, default=16)
p.add_argument("--op", default="add")
p.add_argument("--segments", type=int, default=1)
p.add_argument("--eval_bs", type=int, default=100)
a = p.parse_args()

ta.set_sizes(getattr(a, 'op', 'add'))
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = Tokenizer(10)
P, T, pm, PP, TP, PS, TS, _ = ta.load(a.n, a.digits, 40_000 + a.digits, True, a)

m = Transformer(len(tok), 384, 6, 6, a.pe, causal=(a.mode == "ar"),
                max_len=ta.MAX_PROMPT + ta.CANVAS + a.max_offset + ta.MAX_TEST + 8).to(dev)
m.load_state_dict(torch.load(a.ckpt, map_location=dev)); m.eval()

def show(row):
    return " ".join("__" if t == tok.mask else ("." if t == tok.pad else tok.itos[t]) for t in row)

with torch.no_grad():
    f = ta.sample_diffusion if a.mode == "diff" else ta.sample_ar
    pred = f(m, P.to(dev), pm, PP, TP, PS, TS, tok, a, dev)

    # teacher-forced check: feed the FULLY CORRECT answer with one slot masked.
    # If this is right but sampling is wrong, the model is fine and the sampler
    # is at fault.
    x = T.clone().to(dev)
    x[:, 0] = tok.mask
    logits = m(torch.cat([P.to(dev), x], 1), ta.pad_mask_of(pm, ta.CANVAS, dev),
               ta.pos_for(PP, TP, a, dev), ta.seg_for(PS, TS, a, dev))[:, P.shape[1]:]
    tf = logits.argmax(-1)

for i in range(a.n):
    print(f"\n--- ex {i} ---")
    print("  prompt   :", " ".join(tok.itos[t] for t in P[i].tolist() if t != tok.pad))
    print("  pos ids  :", " ".join(str(x) for x in PP[i].tolist() if x) ,
          "| target pos:", " ".join(str(x) for x in TP[i].tolist() if x))
    print("  gold     :", show(T[i].tolist()))
    print("  sampled  :", show(pred[i].tolist()))
    print("  1-mask TF:", show(tf[i].tolist()))
