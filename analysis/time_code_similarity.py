"""How similar are adjacent seconds in the time code, against distant ones?

THE QUESTION. CLAUDE.md justifies Fourier over a lookup table with one claim:
"delta=3s and delta=4s stay close, so the model can learn a smooth recency prior
instead of 30 unrelated vectors." That is an argument, not a measurement -- nothing
in the repo has ever checked how close adjacent seconds actually are, and
--time_encoding_scale, the single knob that controls it, has been 10.0 in all 69
fourier runs on disk despite CLAUDE.md itself saying to ablate {2, 10, 30}.

This is the temporal twin of smoke_position_encoding.py test 5, which measures the
same property for the SPATIAL code (axis_sum pinned at a near/far ratio of 0.570,
its structural floor; fourier 0.408).

WHAT SCALE ACTUALLY DOES, and why the sweep matters. delta is always normalised to
[0,1) whatever the scale, so slots are 1/T apart -- 0.0333 at T=30. The projection
is delta * w with w ~ N(0, scale), so the argument gap between ADJACENT slots is
about scale/T radians and between the window's ENDS about scale radians:

    scale=2   adjacent 0.07 rad, ends 1.9 rad   -> everything correlated, 30 slots
                                                   collapse toward one vector
    scale=10  adjacent 0.33 rad, ends 9.7 rad   -> ends decorrelate (>2pi)
    scale=30  adjacent 1.0 rad,  ends 29 rad    -> adjacent already far apart, and
                                                   the ends have wrapped many times,
                                                   so distant pairs ALIAS back to
                                                   looking similar

Aliasing is the failure the monotonicity check below catches: a code whose
similarity stops decreasing, or comes back up, cannot express "older" past that gap.

READING THE OUTPUT.
  cos by gap            the curve. Monotone decay is what a usable recency code
                        looks like.
  near/far ratio        mean ||T_i - T_j|| at gap 1, over the same at the largest
                        gap. 1.0 means adjacent slots are as different as the window's
                        two ends -- no smoothness at all, which is exactly where an
                        untrained lookup table sits, and is the cost MMT pays.
  first gap cos<0.5     how many seconds the code can actually tell apart.
  norm                  the amplitude, for reference against the tokens it is added
                        to: a location/gaze/obb token is ~48 under
                        POSITION_ENCODING=fourier and ~27.7 under the index tables,
                        a visual token ~5.9. 'fourier' + gain='fixed' returns 0.707.

Usage:
    python analysis/time_code_similarity.py                    # scale sweep, at init
    python analysis/time_code_similarity.py --checkpoint <dir> # + the TRAINED code
    python analysis/time_code_similarity.py --scales 2,5,10,20,30 --obs_time 30
"""
import argparse
import glob
import math
import os

import numpy as np

H = 768          # transformer_hidden_size
GRID_DEFAULT = 30


def fourier_code(obs_time, scale, seed, hidden=H):
    """Exactly model._time_encoding's fourier branch, before the gain."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, scale, size=(hidden // 2, 1))
    delta = (obs_time - 1 - np.arange(obs_time, dtype=np.float64)) / obs_time
    proj = delta[:, None] @ w.T                       # [T, H/2]
    return np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)   # [T, H]


def sinusoidal_code(obs_time, max_wavelength=10000.0, hidden=H):
    """Vaswani et al. 2017's sinusoidal encoding -- NOT what this repo calls 'fourier'.

    The difference is the spectrum, and it is the whole point. 'fourier' draws its
    frequencies from N(0, scale), which has ONE characteristic scale, so the code has
    one resolution and scale trades adjacent-second sharpness against window-length
    range. This is a fixed GEOMETRIC ladder -- wavelength 2*pi up to
    2*pi*max_wavelength -- so different dimensions resolve different time scales at
    the same time, which is what a multi-scale code buys and a Gaussian one cannot.

    max_wavelength=10000 is the literature default and is tuned for sequences of
    thousands. Over T=30 slots most of that ladder is dead: a dimension whose
    wavelength is 10^4 steps is constant across 30 of them and carries nothing. Pass
    something on the order of the window (2*T) to spend the whole ladder on the range
    that exists here -- both are reported below because the gap between them is the
    cost of using the default unexamined.
    """
    pos = np.arange(obs_time, dtype=np.float64)[:, None]
    i = np.arange(hidden // 2, dtype=np.float64)[None, :]
    inv_freq = 1.0 / (max_wavelength ** (2 * i / hidden))
    ang = pos * inv_freq
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)


def lookup_code(obs_time, seed, hidden=H):
    """nn.Embedding(T, H) at its default init: N(0,1) per element."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(obs_time, hidden))


def report(name, code, note=''):
    """Similarity-by-gap, the near/far ratio, resolution and an aliasing check."""
    T = code.shape[0]
    norms = np.linalg.norm(code, axis=1)
    unit = code / norms[:, None]
    cos = unit @ unit.T
    dist = np.linalg.norm(code[:, None, :] - code[None, :, :], axis=-1)

    gaps = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    by_gap_cos = np.array([cos[gaps == g].mean() for g in range(1, T)])
    by_gap_dist = np.array([dist[gaps == g].mean() for g in range(1, T)])

    ratio = by_gap_dist[0] / by_gap_dist[-1]
    below = np.nonzero(by_gap_cos < 0.5)[0]
    first_half = (below[0] + 1) if len(below) else None

    # NOISE FLOOR. cos here is an average over hidden/2 frequency terms (or, for the
    # lookup, a dot product of two random hidden-dim vectors), so its sampling error
    # is ~1/sqrt(hidden/2) = 0.051 at H=768. A curve wandering inside +-2x that is at
    # zero, not structured -- an earlier version of this script flagged every setting
    # as "aliasing" because it tested strict monotonicity against exactly that
    # jitter. Real aliasing is a REBOUND: similarity climbing back to a value the
    # noise floor cannot explain, which means the code cannot express "older" past
    # that gap.
    floor = 2.0 / math.sqrt(code.shape[1] / 2)
    argmin = int(by_gap_cos.argmin())
    # Aliasing is a RISE after the minimum, not a high minimum. Comparing the
    # post-minimum max against the floor absolutely (the previous version) flags any
    # monotone curve that plateaus above it -- e.g. sinusoidal at the 10000 default,
    # which decays 0.973 -> 0.567 without ever turning back up.
    rise = float(by_gap_cos[argmin:].max() - by_gap_cos[argmin])
    aliasing = rise > floor
    # How much of the similarity range the code actually uses. "informative to gap N"
    # can be true while the curve only travels from 0.97 to 0.57 -- technically
    # ordered, but every pair is still mostly the same vector.
    dyn = float(by_gap_cos.max() - by_gap_cos.min())
    # Where the curve stops carrying information: the last gap whose |cos| still
    # clears the floor. Beyond it every pair looks equally dissimilar, so the code
    # cannot order them.
    informative = np.nonzero(np.abs(by_gap_cos) > floor)[0]
    horizon = (informative[-1] + 1) if len(informative) else 0

    print(f"\n=== {name} ===" + (f"   {note}" if note else ''))
    print(f"  raw row norm        {norms.mean():.3f}"
          f"   -> {norms.mean() / math.sqrt(H):.3f} under gain=fixed, "
          f"or x the trained scalar under gain=learned")
    show = [g for g in (1, 2, 3, 5, 10, 15, 20, 25, T - 1) if 1 <= g <= T - 1]
    print("  cos by gap         " + '  '.join(f"g{g}={by_gap_cos[g - 1]:+.3f}" for g in show))
    print(f"  slot 5 vs 6         {cos[5, 6]:+.4f}   (adjacent, mid-window)")
    print(f"  slot 0 vs {T - 1:<2}        {cos[0, T - 1]:+.4f}   (the window's two ends)")
    print(f"  near/far ratio      {ratio:.4f}   "
          f"(1.0 = adjacent as different as the ends; 0 = perfectly smooth)")
    print(f"  first gap cos<0.5   {first_half if first_half else '(never)'}"
          f"        <- seconds the code can tell apart")
    print(f"  informative to gap  {horizon}"
          f"        <- beyond this every pair is equally dissimilar (floor {floor:.3f})")
    print(f"  dynamic range       {dyn:.3f}"
          f"    <- cos spread used; small = every pair still near-identical")
    print(f"  aliasing            {'YES, rises ' + format(rise, '.3f') + ' after gap ' + str(argmin + 1)}"
          if aliasing else "  aliasing            no")
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='2,5,10,20,30')
    ap.add_argument('--obs_time', type=int, default=GRID_DEFAULT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoint', default='',
                    help="A checkpoint dir, or a snapshot dir (its last checkpoint is "
                         "used). Reports the TRAINED time_W_r / time_table instead of "
                         "an init draw -- which is the number that matters, since "
                         "trainer.py puts time_gain (not W_r) in the fast group and "
                         "W_r itself was measured moving only 0.025% over 19k steps.")
    args = ap.parse_args()
    T = args.obs_time

    print(f"observation_time = {T} slots; slot n holds delta = ({T}-1-n)/{T}, so the "
          f"LAST slot is\nthe second nearest the anchor. Gap g = |slot_i - slot_j| "
          f"= g seconds apart.")

    for s in [float(x) for x in args.scales.split(',')]:
        report(f"fourier, scale={s:g} (init)", fourier_code(T, s, args.seed),
               note='<- the shipped default' if s == 10.0 else '')

    report(f"sinusoidal, max_wavelength=10000 (Vaswani default)",
           sinusoidal_code(T, 10000.0), note='most of the ladder is dead over 30 slots')
    report(f"sinusoidal, max_wavelength={2 * T} (= 2T, fitted to the window)",
           sinusoidal_code(T, float(2 * T)), note='the ladder spent on the range that exists')

    report("lookup / MMT T_1..T_D (init)", lookup_code(T, args.seed),
           note='no smoothness by construction')

    if args.checkpoint:
        ck = os.path.abspath(args.checkpoint)
        if not os.path.exists(os.path.join(ck, 'model.safetensors')):
            cks = glob.glob(f"{ck}/results/*/checkpoint-*")
            if not cks:
                raise SystemExit(f"no model.safetensors and no checkpoints under {ck}")
            ck = max(cks, key=lambda c: int(c.rsplit('-', 1)[1]))
        from safetensors.numpy import load_file
        w = load_file(f"{ck}/model.safetensors")
        print(f"\ntrained checkpoint: {os.path.basename(ck)}")
        kw = [k for k in w if k.endswith('time_W_r')]
        kt = [k for k in w if k.endswith('time_table.weight')]
        kg = [k for k in w if k.endswith('time_gain')]
        gain = float(w[kg[0]]) if kg else None
        if kw:
            W = w[kw[0]].astype(np.float64)
            delta = (T - 1 - np.arange(T, dtype=np.float64)) / T
            proj = delta[:, None] @ W.T
            code = np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)
            amp = (f"gain=learned, trained value {gain:.3f}" if gain is not None
                   else "gain=fixed, so the code ships at norm 0.707")
            report("fourier, TRAINED W_r", code, note=amp)
            print(f"  W_r std             {W.std():.4f}   "
                  f"(init was --time_encoding_scale)")
        elif kt:
            report("lookup, TRAINED table", w[kt[0]].astype(np.float64))
        else:
            print("  this checkpoint has no time code (time_encoding='none')")


if __name__ == '__main__':
    main()