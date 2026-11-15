"""Checks for --token_layernorm and --time_encoding_gain.

Run: python smoke_token_scale.py       (pure torch, no dataset needed, ~30s on CPU)

Both flags exist because of one measurement. On the trained te10 checkpoint
(2026-08-12 LOOCV, --time_encoding fourier --time_encoding_scale 10) the vectors
entering the encoder had these norms:

    time code   0.707      video token    5.88      location token  27.71
    cls_token   0.56       obb token     66.89      sep token       27.15

Nothing chose those numbers. nn.Embedding inits N(0,1), so one row is sqrt(768) =
27.7 -- which is exactly a location or gaze token, since those are fed as three
separate one-row streams -- while an obb token sums five rows and lands at 66.9.
nn.Linear inits U(+-1/sqrt(fan_in)) with fan_in=4096, so visual_mapper's output is
an order of magnitude quieter than the segments it competes with. cls_token uses
std=0.02 and is quieter still -- and it is the readout seat. Since
nn.TransformerEncoderLayer is post-LN, layer 1's attention sees exactly this.

The time code is the worst case and the one that cannot fix itself: cos/sin are
bounded, so W_r sets FREQUENCIES and never amplitude. No gradient to W_r can make
the encoding louder, and measurement agrees -- W_r moved 0.025% in 19,000 steps
while the te10 sweep's mean best PR-AUC (.1288) matched the no-encoding runs
(.1301). So the te10 experiment did not test recency; it tested silence.

  * token_layernorm puts every segment at sqrt(D).
  * time_encoding_gain='learned' unit-norms the code and multiplies by a trainable
    scalar, giving the model an amplitude to choose.

Test 5 is the one that matters, and its expected size is not a guess. The effect of
reordering the seconds is linear in the code's SHARE of the token it rides on, so
the ratio between the two arms is predictable: 0.707/27.71 = 2.6% under the legacy
scaling against 8.0/27.71 = 28.9% at the default gain on LayerNorm'd tokens, i.e.
11.3x. The band is 8-16x, and a ratio far ABOVE it would be as suspicious as one
below -- it would mean something other than the intended amplitude changed.

Test 7 covers the other half of "learnable": AdamW moves a scalar by ~lr per step,
so at the production lr=1e-5 time_gain could travel ~0.1 over a whole run and the
init would silently BE the experiment. trainer.py gives it its own lr=1e-3 group.
"""
import math
import tempfile
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CustomConfig, CustomTransformerModel  # noqa: E402
from object_class_vocab import NUM_OBJECT_CLASSES  # noqa: E402
from smoke_decoder import B, N_OBB, NUM_VOXELS, T, make_batch  # noqa: E402

MODALITY = 'video,location,obb,gaze'
D = 768
SQRT_D = math.sqrt(D)
FAILS = []


def check(ok, label, detail=''):
    print(f"   {label:<58} {detail:<26} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(label)
    return ok


def build(seed=0, **overrides):
    base = dict(num_object_classes=NUM_OBJECT_CLASSES, num_transformer_layers=2,
                use_static_env=True, max_obb_tokens=N_OBB, input_modality=MODALITY,
                readout='cls', decoder='voxel_query')
    base.update(overrides)
    torch.manual_seed(seed)
    return CustomTransformerModel(CustomConfig(**base)).eval()


def logits(model, batch):
    with torch.no_grad():
        return model(**batch)['interaction']


def segment_norms(model, batch):
    """Mean token norm per segment, read off the tensor the encoder actually
    receives. Hooked rather than recomputed so this cannot drift from forward()."""
    captured = {}

    def hook(_mod, args, _kwargs):
        captured['seq'] = args[0].detach()          # [L, B, D]

    handle = model.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        out = model(**batch, need_attention=True)
    handle.remove()
    seq = captured['seq'].permute(1, 0, 2)          # [B, L, D]
    return {name: seq[:, s:e].norm(dim=-1).mean().item()
            for name, s, e in out['segments']}


def reverse_seconds(batch, keys=('person_locations', 'gaze_voxels')):
    b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    for key in keys:
        b[key] = torch.flip(b[key], dims=[1])
    return b


batch = make_batch(0)

print("1. OFF IS OFF -- both flags default to the pre-change model")
m_ref = build(0, time_encoding='fourier')                  # neither flag mentioned
m_off = build(0, time_encoding='fourier',
              token_layernorm=False, time_encoding_gain='fixed')
check(torch.equal(logits(m_ref, batch), logits(m_off, batch)),
      "explicit defaults == omitting both flags")
names_ref = {n for n, _ in m_ref.named_parameters()}
names_off = {n for n, _ in m_off.named_parameters()}
check(names_ref == names_off, "no parameters added by the defaults",
      f"{len(names_ref)} params")
check(not any(n.startswith('token_ln') or n == 'time_gain' for n in names_ref),
      "token_ln / time_gain absent when off")

print("\n2. THE MEASUREMENT THAT MOTIVATED THIS -- segment scales without the flag")
base_norms = segment_norms(m_ref, batch)
for name in ('cls', 'video', 'location', 'obb'):
    print(f"   {name:<58} {base_norms[name]:<26.2f}")
spread = max(base_norms.values()) / min(base_norms.values())
check(spread > 10, "segments differ by more than 10x untouched",
      f"{spread:.1f}x spread")

print("\n3. token_layernorm PUTS EVERY SEGMENT AT sqrt(D)")
m_ln = build(0, time_encoding='fourier', token_layernorm=True)
ln_norms = segment_norms(m_ln, batch)
for name, value in sorted(ln_norms.items()):
    # 'sep' and 'cls' are single vectors: LayerNorm makes them exactly sqrt(D).
    # The rest carry the time code on top, so they sit a hair above it.
    ok = abs(value - SQRT_D) / SQRT_D < 0.05
    check(ok, f"{name} token norm ~= sqrt(768) = 27.71", f"{value:.2f}")
n_added = len({n for n, _ in m_ln.named_parameters()} - names_ref)
check(n_added == 8 * 2, "one LayerNorm (gamma+beta) per segment type",
      f"{n_added} tensors, 12288 params")

print("\n4. time_encoding_gain='learned' GIVES THE CODE AN AMPLITUDE")
m_gain = build(0, time_encoding='fourier', time_encoding_gain='learned',
               time_encoding_gain_init=4.0)
code_fixed = m_ref._time_encoding(T, torch.device('cpu'), torch.float32)
code_gain = m_gain._time_encoding(T, torch.device('cpu'), torch.float32)
check(abs(code_fixed.norm(dim=1).mean().item() - 0.707) < 0.01,
      "legacy code norm is 0.707 (2.6% of a location token)",
      f"{code_fixed.norm(dim=1).mean():.4f}")
check(abs(code_gain.norm(dim=1).mean().item() - 4.0) < 0.01,
      "learned code norm == the gain, exactly",
      f"{code_gain.norm(dim=1).mean():.4f}")
try:
    build(0, time_encoding='none', time_encoding_gain='learned')
    check(False, "gain without an encoding must fail loudly")
except ValueError:
    check(True, "gain without an encoding raises rather than idling")

m_train = build(0, time_encoding='fourier', time_encoding_gain='learned',
                token_layernorm=True).train()
out = m_train(**batch, labels={'interaction': torch.zeros(B, NUM_VOXELS)})
out['loss'].backward()
g = m_train.time_gain.grad
check(g is not None and g.abs().item() > 0, "time_gain receives gradient",
      f"|grad| = {g.abs().item():.6f}")
# All eight are built regardless of modality, so a checkpoint's state_dict does not
# depend on which segments were active. The four unused here (env, trajectory) must
# therefore receive NO gradient -- a non-None grad on those would mean an inactive
# segment is reaching the sequence.
active_ln = ('video', 'location', 'obb', 'gaze', 'sep', 'cls')
check(all(m_train.token_ln[k].weight.grad is not None
          and m_train.token_ln[k].weight.grad.abs().sum() > 0 for k in active_ln),
      "every ACTIVE segment's LayerNorm receives gradient",
      f"{len(active_ln)} of 8")
check(all(m_train.token_ln[k].weight.grad is None for k in ('env', 'trajectory')),
      "inactive segments' LayerNorms receive none")

print("\n5. THE POINT -- recency must now actually reach the logits")
rev = reverse_seconds(batch)
d_fixed = (logits(m_ref, batch) - logits(m_ref, rev)).abs().max().item()
d_both = (logits(build(0, time_encoding='fourier', time_encoding_gain='learned',
                       token_layernorm=True), batch)
          - logits(build(0, time_encoding='fourier', time_encoding_gain='learned',
                         token_layernorm=True), rev)).abs().max().item()
print(f"   reversing location+gaze, fixed gain                        max|delta| = {d_fixed:.4e}")
print(f"   reversing location+gaze, learned gain + token LN           max|delta| = {d_both:.4e}")
# 28.9% / 2.6% = 11.3x predicted at the default gain init of 8.0; see the module
# docstring for why this is the number to expect rather than "as large as possible".
check(8 < d_both / d_fixed < 16, "time order moves the logits ~11.3x more (predicted)",
      f"{d_both / d_fixed:.1f}x")

print("\n6. NOTHING ELSE BROKE")
m_pad = build(0, time_encoding='fourier', time_encoding_gain='learned',
              token_layernorm=True)
b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
# Padded OBB slots must stay inert: scribble on them and the logits must not move.
b2['obb_classes'][:, -3:] = 7
b2['obb_volumes'][:, -3:] = 5
check(torch.allclose(logits(m_pad, batch), logits(m_pad, b2), atol=1e-5),
      "padded OBB slots still never reach the output",
      f"max|delta| = {(logits(m_pad, batch) - logits(m_pad, b2)).abs().max():.2e}")


with tempfile.TemporaryDirectory() as d:
    m_pad.save_pretrained(d)
    reloaded = CustomTransformerModel.from_pretrained(d).eval()
    check(torch.allclose(logits(m_pad, batch), logits(reloaded, batch), atol=1e-5),
          "checkpoint round-trips with both flags set")
    check(reloaded.config.token_layernorm and
          reloaded.config.time_encoding_gain == 'learned',
          "config survives save/load")

print("\n7. time_gain GETS ITS OWN LEARNING RATE")
# Without this the flag is decorative: AdamW moves a scalar by ~lr per step, so at
# the production lr=1e-5 over 19.4k steps with linear decay time_gain could travel
# ~0.1 in total and the init would silently BE the experiment. See
# trainer.create_optimizer.
from transformers import TrainingArguments  # noqa: E402

from trainer import FAST_LR, FAST_PARAMS, CustomTrainer  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    targs = TrainingArguments(output_dir=d, learning_rate=1e-5, report_to='none')

    def groups_for(model):
        t = CustomTrainer(model=model, args=targs)
        t.create_optimizer()
        return t.optimizer.param_groups, t

    gain_groups, t_gain = groups_for(build(0, time_encoding='fourier',
                                           time_encoding_gain='learned',
                                           token_layernorm=True))
    fast = [g for g in gain_groups if g.get('lr') == FAST_LR]
    # Count the gains this build ACTUALLY has, not len(FAST_PARAMS): that set is a
    # candidate list and each entry is created under its own config flag, so a model
    # normally holds only one or two of them. Asserting against the set's size broke
    # the moment 'class_gain' was added.
    built = {n for n, _ in t_gain.model.named_parameters()} & FAST_PARAMS
    check(len(fast) == 1 and len(fast[0]['params']) == len(built),
          f"the gains this build has ({sorted(built)}) sit alone at lr={FAST_LR}",
          f"{len(fast[0]['params']) if fast else 0} param(s) for {len(built)} gain(s)")
    check(fast and fast[0]['weight_decay'] == 0.0,
          "and is not weight-decayed toward silence")
    n_grouped = sum(len(g['params']) for g in gain_groups)
    n_train = sum(1 for p in t_gain.model.parameters() if p.requires_grad)
    check(n_grouped == n_train, "every trainable parameter is in exactly one group",
          f"{n_grouped} of {n_train}")

    default_groups, _ = groups_for(build(0, time_encoding='fourier'))
    check(not any(g.get('lr') == FAST_LR for g in default_groups),
          "the default config builds no fast group at all",
          f"{len(default_groups)} groups")
    # Travel budget = sum of per-step lr under linear decay = lr*T/2.
    budget = FAST_LR * 19407 / 2
    check(budget > 5, "budget spans silent-to-audible over a production run",
          f"~{budget:.1f} from an init of 8.0")

print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ALL TOKEN-SCALE CHECKS PASSED")
