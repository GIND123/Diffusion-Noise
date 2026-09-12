"""Star-graph task (Bachmann & Nagarajan, ICML 2024) + tokenizer.

A star graph has `deg` arms of `path_len` edges radiating from a center node.
The model sees the shuffled edge list plus "center,goal" and must emit the path
from center to goal.

AR models fail here: emitting the first path node requires knowing which arm
reaches the goal, but teacher forcing lets the model learn the trivial
copy-the-rest rule instead, so at inference it guesses (~1/deg accuracy).
"""
import numpy as np

SPECIALS = ["[PAD]", "[MASK]", "|", ",", "=", "[EOS]", "+"]
DIGITS = "0123456789abcdef"   # up to base 16


class Tokenizer:
    def __init__(self, num_nodes=50):
        self.itos = SPECIALS + ([DIGITS[i] for i in range(num_nodes)]
                                if num_nodes <= 16 else [str(i) for i in range(num_nodes)])
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad = self.stoi["[PAD]"]
        self.mask = self.stoi["[MASK]"]
        self.eos = self.stoi["[EOS]"]

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi[t] for t in toks]


def make_example(rng, num_nodes, deg, path_len):
    """Returns (prompt_tokens, target_tokens) as lists of string tokens."""
    nodes = rng.choice(num_nodes, size=deg * path_len + 1, replace=False)
    center, rest = nodes[0], nodes[1:]
    arms = rest.reshape(deg, path_len)

    edges = []
    for a in range(deg):
        prev = center
        for j in range(path_len):
            edges.append((prev, arms[a, j]))
            prev = arms[a, j]

    goal_arm = rng.integers(deg)
    goal = arms[goal_arm, -1]
    path = [center] + list(arms[goal_arm])

    order = rng.permutation(len(edges))
    prompt = []
    for k in order:
        u, v = edges[k]
        prompt += [str(u), ",", str(v), "|"]
    prompt += [str(center), ",", str(goal), "="]

    target = []
    for i, n in enumerate(path):
        target.append(str(n))
        if i < len(path) - 1:
            target.append(",")
    target.append("[EOS]")
    return prompt, target


def build_dataset(n, num_nodes, deg, path_len, max_prompt, canvas, seed=0):
    """Fixed-size arrays. Canvas is padded so longer OOD paths remain
    representable by a fixed-canvas diffusion model (tests H1)."""
    tok = Tokenizer(num_nodes)
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)
    for i in range(n):
        p, t = make_example(rng, num_nodes, deg, path_len)
        if len(p) > max_prompt or len(t) > canvas:
            raise ValueError(f"too long: prompt {len(p)}>{max_prompt} or target {len(t)}>{canvas}")
        pe, te = tok.encode(p), tok.encode(t)
        P[i, : len(pe)] = pe
        pmask[i, : len(pe)] = True
        T[i, : len(te)] = te
    return P, T, pmask, tok


def build_sort_dataset(n, num_nodes, k, max_prompt, canvas, seed=0):
    """Negative control: sort k integers. Order-insensitive — every output
    position is computable independently, so decoding order should not matter.
    Also a pipeline correctness probe: if this does not learn, the bug is mine,
    not the task's."""
    tok = Tokenizer(num_nodes)
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)
    for i in range(n):
        vals = rng.choice(num_nodes, size=k, replace=False)
        prompt = []
        for j, v in enumerate(vals):
            prompt.append(str(v))
            prompt.append("," if j < k - 1 else "=")
        target = []
        for j, v in enumerate(sorted(vals)):
            target.append(str(v))
            if j < k - 1:
                target.append(",")
        target.append("[EOS]")
        pe, te = tok.encode(prompt), tok.encode(target)
        P[i, : len(pe)] = pe
        pmask[i, : len(pe)] = True
        T[i, : len(te)] = te
    return P, T, pmask, tok


def build_op_dataset(n, digits, max_prompt, canvas, seed=0, exact=False,
                     reverse=True, max_offset=8, op="add", base=10):
    """Place-value-aligned dataset for addition or multiplication.

    Multiplication is the harder case on purpose: its algorithm is NOT
    place-local (every output digit depends on many input pairs), so it probes
    whether the method works only when place-value alignment happens to match
    the algorithm's dependency structure.
    """
    tok = Tokenizer(16)          # vocabulary covers every base we support
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    PP = np.zeros((n, max_prompt), dtype=np.int64)
    TP = np.zeros((n, canvas), dtype=np.int64)
    PS = np.zeros((n, max_prompt), dtype=np.int64)
    TS = np.zeros((n, canvas), dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)
    sym = {"add": "+", "mul": "|", "sub": ","}[op]   # reuse existing vocab symbols

    for i in range(n):
        d = digits if exact else int(rng.integers(1, digits + 1))
        # Build operands digit-by-digit: 10**20 overflows int64, but Python ints
        # are arbitrary precision, so the arithmetic itself is exact at any size.
        def draw(nd):
            ds = rng.integers(0, base, size=nd)
            if nd > 1 and ds[0] == 0:
                ds[0] = rng.integers(1, base)
            return "".join(DIGITS[int(x)] for x in ds)

        def to_base(v):
            if v == 0:
                return DIGITS[0]
            out = ""
            while v:
                out = DIGITS[v % base] + out
                v //= base
            return out

        pa, pb = draw(d), draw(d)
        if op == "sub" and int(pa, base) < int(pb, base):
            pa, pb = pb, pa            # keep results non-negative
        a, b = int(pa, base), int(pb, base)
        ps = to_base({"add": a + b, "mul": a * b, "sub": a - b}[op])
        off = int(rng.integers(0, max_offset + 1))

        prompt, ppos, pseg = [], [], []
        for j, ch in enumerate(pa):
            prompt.append(ch); ppos.append(off + len(pa) - j); pseg.append(1)
        prompt.append(sym); ppos.append(0); pseg.append(0)
        for j, ch in enumerate(pb):
            prompt.append(ch); ppos.append(off + len(pb) - j); pseg.append(2)
        prompt.append("="); ppos.append(0); pseg.append(0)

        target = list(ps[::-1] if reverse else ps) + ["[EOS]"]

        pe, te = tok.encode(prompt), tok.encode(target)
        if len(pe) > max_prompt or len(te) > canvas:
            raise ValueError(f"too long: {len(pe)}>{max_prompt} or {len(te)}>{canvas}")
        P[i, : len(pe)] = pe;  PP[i, : len(pe)] = ppos;  PS[i, : len(pe)] = pseg
        T[i, : len(te)] = te
        # Every canvas slot gets a place-value id and the answer segment,
        # INCLUDING slots past the end. Assigning ids only to written slots made
        # the count of non-zero ids equal the answer length, which handed the
        # model the answer length for free at generation time (leak found by
        # src/audit.py) and was information the sequential-id baseline never got.
        TP[i, :] = off + np.arange(canvas) + 1
        TS[i, :] = 3
        pmask[i, : len(pe)] = True
    return P, T, pmask, PP, TP, PS, TS, tok


def build_add_dataset_coupled(n, digits, max_prompt, canvas, seed=0, exact=False,
                              reverse=True, max_offset=8):
    """Addition with SIGNIFICANCE-ALIGNED position ids (our method).

    Instead of numbering tokens by sequence index, every digit is numbered by
    its place value, so digits that must be combined share an id:

        4   7   +   8   5   =   1   3   2
        2   1   0   2   1   0   3   2   1

    The rule "combine equal ids, carry into id+1" is then independent of how
    many digits the numbers have, which is what allows extrapolation to lengths
    never trained on. A random offset is added to every id per example so the
    model keys on *relative* place value rather than memorising absolute ids.

    Returns the usual arrays plus position ids for prompt and target.
    """
    tok = Tokenizer(10)
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    PP = np.zeros((n, max_prompt), dtype=np.int64)
    TP = np.zeros((n, canvas), dtype=np.int64)
    PS = np.zeros((n, max_prompt), dtype=np.int64)
    TS = np.zeros((n, canvas), dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)

    for i in range(n):
        d = digits if exact else int(rng.integers(1, digits + 1))
        lo, hi = (10 ** (d - 1), 10 ** d) if d > 1 else (0, 10)
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        s = a + b
        pa, pb, ps = str(a), str(b), str(s)
        off = int(rng.integers(0, max_offset + 1))

        prompt, ppos, pseg = [], [], []
        for j, ch in enumerate(pa):                 # most significant first
            prompt.append(ch); ppos.append(off + len(pa) - j); pseg.append(1)
        prompt.append("+"); ppos.append(0); pseg.append(0)
        for j, ch in enumerate(pb):
            prompt.append(ch); ppos.append(off + len(pb) - j); pseg.append(2)
        prompt.append("="); ppos.append(0); pseg.append(0)

        target, tpos, tseg = [], [], []
        digs = ps[::-1] if reverse else ps
        for j, ch in enumerate(digs):
            target.append(ch)
            sig = (j + 1) if reverse else (len(ps) - j)
            tpos.append(off + sig); tseg.append(3)
        # EOS gets its own place-value id (one past the most significant digit)
        # and its own segment. Giving it id 0 made it indistinguishable from
        # padding, so the model could not tell which slot should terminate.
        target.append("[EOS]"); tpos.append(off + len(ps) + 1); tseg.append(4)

        pe, te = tok.encode(prompt), tok.encode(target)
        if len(pe) > max_prompt or len(te) > canvas:
            raise ValueError(f"too long: {len(pe)}>{max_prompt} or {len(te)}>{canvas}")
        P[i, : len(pe)] = pe;  PP[i, : len(pe)] = ppos;  PS[i, : len(pe)] = pseg
        T[i, : len(te)] = te
        # Every canvas slot gets a place-value id and the answer segment,
        # INCLUDING slots past the end. Assigning ids only to written slots made
        # the count of non-zero ids equal the answer length, which handed the
        # model the answer length for free at generation time (leak found by
        # src/audit.py) and was information the sequential-id baseline never got.
        TP[i, :] = off + np.arange(canvas) + 1
        TS[i, :] = 3
        pmask[i, : len(pe)] = True
    return P, T, pmask, PP, TP, PS, TS, tok


def build_add_dataset(n, digits, max_prompt, canvas, seed=0, exact=False, reverse=True):
    """Multi-digit addition: "3 7 5 + 2 4 8 =" -> sum digits.

    Output is least-significant-digit-first by default, the standard format in
    the arithmetic length-generalization literature; it removes AR's need to
    know the carry before emitting the leading digit, so the AR baseline is the
    strong version rather than a straw man. Both architectures get the same
    format.

    exact=False samples 1..digits (mixed lengths, for training);
    exact=True fixes the operand length (for a clean eval split).
    """
    tok = Tokenizer(10)
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)
    for i in range(n):
        d = digits if exact else int(rng.integers(1, digits + 1))
        a = int(rng.integers(10 ** (d - 1), 10 ** d)) if d > 1 else int(rng.integers(0, 10))
        b = int(rng.integers(10 ** (d - 1), 10 ** d)) if d > 1 else int(rng.integers(0, 10))
        s = a + b
        pa, pb, ps = str(a), str(b), str(s)
        if reverse:
            ps = ps[::-1]
        prompt = list(pa) + ["+"] + list(pb) + ["="]
        target = list(ps) + ["[EOS]"]
        pe, te = tok.encode(prompt), tok.encode(target)
        if len(pe) > max_prompt or len(te) > canvas:
            raise ValueError(f"too long: {len(pe)}>{max_prompt} or {len(te)}>{canvas}")
        P[i, : len(pe)] = pe
        pmask[i, : len(pe)] = True
        T[i, : len(te)] = te
    return P, T, pmask, tok


def target_lens(deg, path_len):
    return 2 * (path_len + 1)  # nodes + separators + EOS


def build_seq_dataset(n, length, max_prompt, canvas, seed=0, exact=False,
                      max_offset=8, task="parity"):
    """Non-arithmetic length-generalization probes.

    These separate "does the method need a CHAIN dependency" from "does it need
    place value" - the question multiplication raised but could not answer:

      parity  : output[j] = XOR of input[0..j].  Sequential chain, exactly like
                a carry, but no arithmetic and no place value.
      reverse : output[j] = input[n-1-j].        Positional alignment, NO chain.

    Aligned ids point each output slot at the input slot it depends on, which is
    the direct analogue of place-value alignment for addition.
    """
    tok = Tokenizer(10)
    rng = np.random.default_rng(seed)
    P = np.full((n, max_prompt), tok.pad, dtype=np.int64)
    T = np.full((n, canvas), tok.pad, dtype=np.int64)
    PP = np.zeros((n, max_prompt), dtype=np.int64)
    TP = np.zeros((n, canvas), dtype=np.int64)
    PS = np.zeros((n, max_prompt), dtype=np.int64)
    TS = np.zeros((n, canvas), dtype=np.int64)
    pmask = np.zeros((n, max_prompt), dtype=bool)

    for i in range(n):
        L = length if exact else int(rng.integers(1, length + 1))
        off = int(rng.integers(0, max_offset + 1))
        hi = 2 if task == "parity" else 10
        xs = [int(v) for v in rng.integers(0, hi, size=L)]

        if task == "parity":
            run, ys = 0, []
            for v in xs:
                run ^= v
                ys.append(run)
        else:
            ys = list(reversed(xs))

        prompt = [str(v) for v in xs] + ["="]
        ppos = [off + j + 1 for j in range(L)] + [0]
        pseg = [1] * L + [0]

        target = [str(v) for v in ys] + ["[EOS]"]
        # each output slot carries the id of the input slot it depends on
        # reverse counts down, so slots past the sequence end would go negative
        tpos = [(off + j + 1) if task == "parity" else max(0, off + L - j)
                for j in range(canvas)]
        tseg = [3] * canvas

        pe, te = tok.encode(prompt), tok.encode(target)
        if len(pe) > max_prompt or len(te) > canvas:
            raise ValueError(f"too long: {len(pe)}>{max_prompt} or {len(te)}>{canvas}")
        P[i, : len(pe)] = pe;  PP[i, : len(pe)] = ppos;  PS[i, : len(pe)] = pseg
        T[i, : len(te)] = te;  TP[i, :] = tpos;          TS[i, :] = tseg
        pmask[i, : len(pe)] = True
    return P, T, pmask, PP, TP, PS, TS, tok
