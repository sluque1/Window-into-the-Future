# Method 1 — gradient-based modality and token importance

Results for the four checkpoints of the two LOOCV runs. Architecture, token
layout and run identification are in [ARCHITECTURE.md](ARCHITECTURE.md).

**Everything here is from a verified run.** Numbers produced by the two attempts
that failed their own checks are excluded and named in §8 so they are not
mistaken for results later.

---

## 1. What was measured

For every episode of a run's held-out test split, the gradient of a scalar target
with respect to the **token embeddings entering the transformer encoder** — the
`[592, B, 768]` tensor, captured by a `forward_pre_hook` on `model.transformer`
with no change to `model.py`.

Not with respect to parameters. A gradient on `visual_mapper.weight` says the
visual projection is far from optimal, not that video matters to the prediction,
and `spatial_W_r` is shared by location, obb, trajectory **and** the decoder's
4096 queries — one gradient summing four consumers, unsplittable. Attribution at
the token vectors avoids this, because sequence positions are disjoint.

**Primary target** `pos_contrast` = mean logit over ground-truth positive voxels
− mean logit over all 4096. The contrast removes the per-episode offset; the
architecture carries a large static per-voxel field through the decoder's query
residual and bias, neither of which depends on input tokens.

**Scores**, per token: `grad_l2` = ‖∂T/∂x_p‖₂ (magnitude, unsigned);
`gxi` = ⟨∂T/∂x_p, x_p⟩ (signed first-order contribution); plus ‖x_p‖₂ recorded as
a scale control. Aggregated per modality as a **total over valid tokens** and a
**mean per valid token** — reported side by side throughout, because they rank
the modalities differently by construction and reporting one alone misleads.

Padding is excluded from both numerator and denominator. The obb per-token figure
divides by the **real box count** (140.2 on P08, 96.5 on P04), never by the 512
slots.

**Precision.** The runs trained and evaluated with `bf16=True`; this analysis runs
in float32, the dtype the weights are stored in. Measured on 64 episodes:
float32 differs from the saved `predictions.pkl` by mean |Δ| 0.030 (**0.353%** of
a mean |logit| of 8.35) and bfloat16 by 0.001 (**0.014%**) — bf16 lands 26× closer,
confirming the gap is the original run's own precision, not a reconstruction
error. Per-episode Spearman over the 4096 voxels is **0.9998**, so the voxel
ordering PR-AUC reads is unchanged.

---

## 2. Modality importance

Primary target, `grad_l2`, full test splits.

| | A_best | A_final | B_best | B_final |
|---|---|---|---|---|
| held-out participant | P08 | P08 | P04 | P04 |
| episodes | 5007 | 5007 | 3311 | 3311 |
| logged PR-AUC | 0.3941 | 0.3759 | 0.1417 | 0.1337 |
| **video** | **0.6419** | **0.6418** | **0.6440** | **0.6506** |
| obb | 0.2561 | 0.2618 | 0.2261 | 0.2292 |
| location | 0.0672 | 0.0650 | 0.0828 | 0.0783 |
| cls | 0.0151 | 0.0119 | 0.0260 | 0.0219 |
| trajectory | 0.0117 | 0.0123 | 0.0096 | 0.0099 |
| sep | 0.0081 | 0.0072 | 0.0116 | 0.0101 |

Shares sum to 1 across the six rows.

### Per valid token, with the scale control

| segment | valid/ep (A) | ‖x‖ | A_best | A_final | B_best | B_final |
|---|---|---|---|---|---|---|
| cls | 1.0 | 0.56 | 0.03057 | 0.02957 | 0.04722 | 0.05397 |
| **video** | 29.7 | **9.97** | **0.04453** | **0.05488** | **0.04397** | **0.06026** |
| sep | 3.0 | 28.13 | 0.00534 | 0.00586 | 0.00685 | 0.00814 |
| location | 29.7 | 55.41 | 0.00466 | 0.00557 | 0.00547 | 0.00706 |
| obb | 140.2 / 96.5 | 61.43 | 0.00394 | 0.00497 | 0.00489 | 0.00671 |
| trajectory | 5.7 / 4.4 | 62.74 | 0.00437 | 0.00568 | 0.00422 | 0.00589 |

**Video leads on totals *and* per token**, ~11× the other content modalities.
That is unusual — normally one modality wins on totals through token count and
another wins per token. OBB has 96–140 tokens against video's 30 and loses both.

**Not a magnitude artefact.** Video tokens are the *smallest* content tokens
(‖x‖ ≈ 10 against obb's 61), and Spearman(token norm, per-token score) = **−0.77**
on A_best — the ranking runs against size, not with it. `TOKEN_LAYERNORM=false`
in both runs, so nothing equalises the segments and the 110× spread from cls
(0.56) to trajectory (62.7) is set purely by whether a segment was built by
`nn.Linear` or `nn.Embedding`.

### Consistency

Kendall τ over the six segment shares, all pairings:

| pair | share | per token |
|---|---|---|
| A_best vs A_final | +0.867 | +0.867 |
| A_best vs B_best | +0.867 | +0.733 |
| A_best vs B_final | +0.867 | +0.867 |
| A_final vs B_best | +0.733 | +0.600 |
| A_final vs B_final | +0.733 | +0.733 |
| B_best vs B_final | +1.000 | +0.867 |

**τ +0.60 to +1.00 across two kitchens and both training stages — despite PR-AUC
differing 2.8× between the runs.** How the model distributes its sensitivity
transfers across kitchens even though how well it performs does not.

Best vs final: shares nearly identical, but per-token magnitudes uniformly
**higher** in the final checkpoints (video 0.0445 → 0.0549 on A). Sensitivity
grows over training while PR-AUC slightly declines.

---

## 3. Token-level structure

| | A_best | A_final | B_best | B_final |
|---|---|---|---|---|
| video, last second ÷ first | **1.36×** | 1.37× | 1.34× | 1.33× |
| location, last ÷ first | 1.11× | 1.10× | 1.14× | 1.14× |
| obb, slot 0 → slot 150 | 0.0042→0.0030 | 0.0054→0.0038 | 0.0055→0.0028 | 0.0070→0.0040 |

- **Video has a mild, consistent recency gradient** — the second nearest the
  anticipation anchor matters ~1.35× more than the oldest, in all four.
- **Location is flat** (1.10–1.14×): essentially no temporal structure.
- **OBB declines with detection confidence** (~1.4× from rank 0 to 150), so
  higher-confidence boxes matter more.

---

## 4. Channel decomposition

Every token is an exact sum of named parts (`token_layernorm=False` makes `_ln`
the identity; measured `max|token − Σ parts| = 0.0`), and grad×input is bilinear,
so `⟨∇, token⟩ = Σ_parts ⟨∇, part⟩` **exactly**. Verified every batch: channel
reconstruction residual 9.5e-07, channel sum vs gxi total ≤ 2.0e-06.

`pos_code` is one function (`_voxel_pos_embedding`) used by location, obb,
trajectory **and the decoder's 4096 queries**; `class` is shared by obb and
trajectory. "shared" is the fraction of a modality's absolute channel mass
arriving through those.

| checkpoint | segment | total | shared | breakdown |
|---|---|---|---|---|
| A_best | video | +0.5743 | **0.0%** | visual +0.6059, time −0.0316 |
| | location | −0.3723 | 72.9% | pos_code −0.5433, type +0.1864, time −0.0154 |
| | obb | −0.6552 | 53.4% | pos_code −0.3464, type −0.2616, volume −0.0438, **class −0.0035** |
| | trajectory | +0.0227 | 20.0% | type +0.0319, distance −0.0125, pos_code +0.0071 |
| A_final | location | −0.5471 | 83.9% | pos_code −0.6370, type +0.1059 |
| | obb | −0.5538 | 27.8% | type −0.4040, pos_code −0.1547, **class −0.0045** |
| B_best | location | −0.5875 | 91.3% | pos_code −0.5390, type −0.0501 |
| | obb | −0.3829 | 3.5% | type −0.3404, volume −0.0571, **class +0.0003** |
| B_final | location | −0.8503 | 76.0% | pos_code −0.6464, type −0.2015 |
| | obb | −0.2611 | 12.5% | type −0.2803, volume −0.0245, **class +0.0009** |

Three findings:

1. **Video is the only modality with 0% shared parameters.** Its contribution runs
   through `visual_mapper`, which nothing else touches, so it is attributable to
   video alone. Location is 73–91% the shared spatial code — largely a second
   read of what obb, trajectory and the queries also use.
2. **OBB's `class` channel contributes essentially nothing** — |·| ≤ 0.005 against
   segment totals of 0.26–0.66, in all four. Normalised by embedding magnitude it
   is still **12× weaker than position** and 7× weaker than volume (per unit norm:
   pos_code 0.0171, volume 0.0101, type 0.0052, class 0.0014). This bears on the
   OBB-token rationale in CLAUDE.md, which is that OBB tokens exist to preserve
   class distinctions the env grid collapsed.
3. **A large part of OBB's contribution is a constant.** `type_marker` is the same
   vector in every OBB token and carries 40–107% of the segment's signed total —
   so much of what OBB contributes is *how many boxes there are*, not what or
   where they are.

OBB's shared fraction is the least stable quantity here (53% / 28% / 3.5% / 12.5%)
and should not be leaned on.

---

## 5. Integrated Gradients

**Segment-mean baseline, 32 steps.** Completeness — an arithmetic identity, that
the attributions sum to `T(real) − T(baseline)` — holds to a median relative error
of **0.0001–0.0005** (p90 0.0004–0.0051). The IG sum matches the measured delta to
four decimals (−1.2860 vs −1.2859 on A_best).

| checkpoint | T(real) | T(generic) | difference | video | obb | location | trajectory |
|---|---|---|---|---|---|---|---|
| A_best | +6.3726 | +7.6585 | **−1.2859** | −0.9559 | −0.2597 | −0.0586 | −0.0118 |
| A_final | +6.5770 | +7.7151 | **−1.1380** | −0.9528 | −0.1668 | −0.0245 | +0.0060 |
| B_best | +4.4438 | +4.5843 | **−0.1404** | −0.0428 | +0.0335 | −0.1221 | −0.0092 |
| B_final | +4.8481 | +5.1085 | **−0.2604** | −0.1969 | +0.0417 | −0.0820 | −0.0233 |

`T(generic)` is the model fed **one average token per segment** — no episode
content, just "a typical video second, a typical box, a typical position".

**T(real) < T(generic) in all four.** The episode's actual content makes the
ground-truth voxels stand out *less* than a content-free input does. Both remain
strongly positive, so GT voxels outscore the average map either way. This is
consistent with the static-map dominance already recorded in CLAUDE.md.

### IG disagrees with grad×input on video's sign, in all four

| | grad×input | IG |
|---|---|---|
| A_best video | **+0.9773** | **−0.9559** |
| A_final video | +0.8409 | −0.9528 |
| B_best video | +0.2434 | −0.0428 |
| B_final video | +0.1022 | −0.1969 |

**Not a contradiction — different counterfactuals.** `grad×input` linearises
around a *zero* baseline ("what if this token were deleted"); IG here uses the
*segment-mean* baseline ("what if this token were typical"). Video content is
better than nothing but worse than average. Both can hold.

### Stability

The **magnitude** ranking is stable across kitchens (§2, τ ≥ +0.60). The **signed**
ranking is not: A attributes the contrast reduction mostly to video
(−0.96 of −1.29), B mostly to location (−0.12 of −0.14), and obb flips sign
between runs. Report the first as a finding, the second as run-specific.

### Caveat

The generic input is itself out of distribution — 30 identical video tokens is not
something the model ever saw — so "T(generic) is high" may partly reflect feeding
a degenerate input rather than the prior genuinely being better. Method 2's
ablation is the test.

---

## 6. Validation

### Passed

| check | result |
|---|---|
| padded tokens have zero gradient | **exactly 0** across every split |
| batched backward = per-sample backward | max\|Δ\| 4.6e-08 – 4.8e-07 |
| channel reconstruction | residual 9.5e-07 |
| channel sums = gxi totals | ≤ 2.0e-06 |
| IG completeness | median 0.0001–0.0005 |
| logits reproduce `predictions.pkl` | 0.35% (bf16, §1) |
| **untrained-weights control** | τ **+0.60 to +0.73**, max Δshare **0.20–0.28** |
| model in eval mode with autograd live | asserted |

The weights control is the one that matters: attributing an untrained model built
from the same config shifts the ranking substantially, so the scores do read the
trained model.

### Failed — and it is a result

**The label-randomisation control failed.** Attributing towards *another episode's*
ground truth left the modality shares essentially unchanged:

| checkpoint | Kendall τ vs main | max Δshare |
|---|---|---|
| A_best | +1.000 | 0.0031 |
| A_final | +0.867 | 0.0026 |
| B_best | +1.000 | 0.0051 |
| B_final | +1.000 | 0.0045 |

The shuffle worked — target values changed substantially (per-episode correlation
+0.24 for `pos_contrast`, +0.05 for `loss`) — and the label-independent `top20`
target came out bit-identical, so the machinery is right.

---

## 7. Why the label control fails, measured

`m1_voxel_alignment.py`, 16 episodes, 12 GT + 12 random voxels each.

**Per-voxel gradients are not all alike.** Cosine similarity between
`∂logit_v/∂x` for different voxels: pos-pos 0.4953, rand-rand 0.3535, pos-rand
0.1912 — nowhere near 1.0. The model *does* distinguish voxels.

**But the separation is distance, not GT-ness.** GT voxels cluster (mean pair
distance **2.43** grid cells) while random voxels are scattered (**10.68**), and
each voxel's query is built from its coordinates, so nearby voxels get similar
queries automatically. Matching on distance:

| grid distance | pos-pos | rand-rand |
|---|---|---|
| 0–3 | 0.6110 (n=696) | 0.6089 (n=24) |
| 3–5 | 0.2766 (n=340) | 0.3627 (n=78) |
| 5–7 | 0.1863 (n=20) | 0.3460 (n=105) |

At equal distance GT pairs are **never more similar** — in two of three bands they
are less. Distance-matched summary: pos-pos 0.4953 vs rand-rand 0.5873
(gap **−0.0920**) and vs pos-rand 0.4894 (gap **+0.0059**).

*Caveat:* the overlap is thin (24 rand-rand pairs in the closest band; pos-pos has
no pairs beyond 7 cells). The consistent direction across all three bands is what
carries the conclusion, not the exact gap.

**And the reported statistic is voxel-invariant.** Per-voxel modality share, over
all voxels GT and random alike:

| segment | mean | sd | min | max |
|---|---|---|---|---|
| video | 0.6522 | **0.0290** | 0.5382 | 0.6994 |
| obb | 0.2335 | 0.0188 | 0.1940 | 0.2770 |
| location | 0.0721 | 0.0111 | 0.0344 | 0.0964 |

### Conclusion

Voxel-to-voxel differences in the gradient are driven by **position and
essentially nothing else**. Being a future interaction site adds nothing
measurable. So:

- **The modality ranking in §2 is strengthened**: "video ≈ 64%" is not an average
  that happens to be stable, it holds for **every individual voxel** (sd 0.029).
- **Gradient attribution cannot localise on this architecture** — and not because
  the wrong summary was chosen. There is no target-related signal in the gradients
  for any summary to find. This is the token-level counterpart of the static-map
  finding in CLAUDE.md.

---

## 8. Excluded results, and problems found

**Excluded — produced by runs that failed their own checks:**

- **IG with the zero baseline** (completeness error 0.76–1.01, i.e. accounting for
  0–24% of the change). Diagnosed as quadrature error, not a bug: error falls
  8→512 steps as 0.893 → 0.797 → 0.303 → 0.016, while the segment-mean baseline
  reads 0.0008 at 8 steps. `|T(x) − T(b)|` is 5.66 for zero against 1.28 for
  segment-mean — a 4.4× longer path through a region where attention logits scale
  as α² and the model behaves nothing like it does on real data. Superseded by §5.
- **A 32-episode smoke result** that briefly occupied `A_final`'s output directory,
  caught by the manifest episode-count check and quarantined to `out/_stale/`.
- **The first raw pos/rand cosine comparison** (0.50 vs 0.19) read as evidence that
  GT voxels share a signature. It is confounded by distance; §7 supersedes it.

**Problems found in the pipeline or model:**

- `location_embedding_x/y/z` (39,168 params) and `gaze_type_embedding` receive
  **zero gradient** — dead weight in this configuration, as CLAUDE.md predicts for
  `position_encoding=fourier + location_tokens=joint` without gaze.
- The fused OBB cache is **version 1 while the code is version 2**. Both training
  runs carried the same warning so results reproduce, but it is a latent
  reproducibility hazard; rebuilding the cache would change ~3.9% of OBB token
  class ids.
- Cluster jobs inherit `ulimit -t 600` from the login shell
  (`PropagateResourceLimits=ALL`), which killed the first two submissions.
  `--propagate=NONE` does **not** prevent this; `ulimit -t 1800` inside the script
  does, and 1800 s is a hard ceiling.

---

## 9. Reproduce

```bash
cd /path/to/fiction-hd-epic-v2
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate egovlpv2-pip
ulimit -t 1800; export OMP_NUM_THREADS=4

# layout + strict load, no dataset needed
python analysis/modality_importance/verify_layout.py

# main pass + controls + IG, per run
python analysis/modality_importance/m1_gradients.py --run A --which both \
    --control none,weights,labels --control_n 1000 \
    --ig --ig_baseline segment_mean --ig_steps 32 --ig_episodes 256
python analysis/modality_importance/m1_gradients.py --run B --which both ...   # same flags

# diagnostics
python analysis/modality_importance/m1_ig_diagnose.py --run A --n 16
python analysis/modality_importance/m1_voxel_alignment.py --run A --n 16

# tables and figures
python analysis/modality_importance/m1_report.py
```

Throughput 23–26 ep/s for the gradient pass, 2.8 ep/s for IG at 32 steps, on one
H200. Dataset build 220 s (P04) / 300 s (P08), once per invocation.

Outputs in `out/m1/<tag>/{main,weights,labels}/{scores.npz,ig.npz,manifest.json}`
and `out/m1/_report/`. Each `manifest.json` records the checkpoint, config,
dataset kwargs, diagnostics and environment.

---

## 10. What Method 1 answered, and what it did not

**Answered.**
1. *Which modalities contribute most?* Video ~64% of total sensitivity and ~11×
   the per-token value of any other content modality, stable across four
   checkpoints and two kitchens. OBB second on totals purely through token count
   and last per token. Location 7–8%, mostly the shared spatial code. Trajectory
   ~1%.
2. *Which tokens?* Video shows a mild recency gradient (1.33–1.37×); location is
   flat; OBB declines with detection confidence.
3. *Consistent across checkpoints?* Yes — τ +0.60 to +1.00, including across two
   kitchens whose PR-AUC differs 2.8×.

**Not answered, and provably not answerable this way.** Which modality determines
*where* the interaction goes. Per-voxel gradient differences are purely spatial
(§7), so no gradient-magnitude statistic and no choice of target can recover it.

**Method 2 is the method that can.** Ablation measures the change in PR-AUC, which
is defined against the ground truth, so it cannot be blind to which voxels are the
targets in the way Method 1 is.

**The prediction Method 2 should test.** IG says episode content *reduces* the
GT-vs-average contrast (§5), most of it through video on P08. If removing video
improves held-out PR-AUC, that is a substantial finding. If removing video hurts
badly, the segment-mean counterfactual does not correspond to usefulness — also
worth knowing, and exactly what the cross-method comparison is for.
