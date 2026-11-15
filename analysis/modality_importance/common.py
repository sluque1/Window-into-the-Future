"""Shared machinery for the modality-importance analysis (Methods 1-4).

Nothing here modifies model.py, decoder.py or loader.py. Every hook is a
``register_forward_pre_hook`` that the caller removes again, and the two tensors
they capture are the ones every method needs:

  * ``encoder_input``  [L, B, H] -- the token embeddings entering the encoder,
    AFTER the CLS token is prepended. This is the single point where every
    modality has become one comparable object, and it is what attribution is
    computed with respect to.
  * ``pad_mask``       [B, L] bool, True = ignored -- captured from the DECODER's
    call rather than rebuilt, so it is literally the mask the model used,
    including model.py:1458's all-pad guard (a window with no Boxer boxes has its
    obb slot 0 forced attendable to stop the softmax returning NaN). Rebuilding
    it here from the batch would drift from that.

Why the encoder input is the right attribution point, rather than the raw
per-modality tensors: the modalities arrive as different dtypes and shapes
(float features, int voxel indices, int class ids), so "the gradient with respect
to the input" is not even defined for most of them -- an integer index has no
derivative. At the encoder input they are all 768-dim float vectors on one
sequence, which is what makes video, location, obb and trajectory comparable at
all.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loader import HDEpicInteractionDataset  # noqa: E402
from model import CustomConfig, CustomTransformerModel  # noqa: E402
from train import torch_default_data_collator  # noqa: E402

import runs as runspec  # noqa: E402  (same directory)

OUT_BASE = os.path.join(_HERE, 'out')

# Segment names that are real input modalities, in sequence order. 'cls' and
# 'sep' are reported separately and never folded into a modality: cls is a
# learned summary seat with no input content, and the three sep positions are one
# shared parameter repeated, not three tokens.
MODALITIES = ('video', 'location', 'obb', 'trajectory')
STRUCTURAL = ('cls', 'sep')


# ---------------------------------------------------------------------------
# checkpoint + dataset
# ---------------------------------------------------------------------------

def load_checkpoint(path, device='cpu'):
    """Load a checkpoint into a model built from its own config.json.

    Strict in both directions. Every failure mode here produces plausible numbers
    if ignored -- a partially matching state dict leaves randomly initialised
    parameters that still emit a tidy attribution map -- so a non-empty missing or
    unexpected list is fatal, and strict=False is never used to paper over one.
    """
    if not os.path.isdir(path):
        raise SystemExit(f"no such checkpoint directory: {path}")
    cfg_path = os.path.join(path, 'config.json')
    if not os.path.exists(cfg_path):
        raise SystemExit(f"{path} has no config.json -- is it a checkpoint directory?")
    config = CustomConfig.from_pretrained(path)

    weights = None
    for name in ('model.safetensors', 'pytorch_model.bin'):
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            weights = candidate
            break
    if weights is None:
        raise SystemExit(f"{path} holds no model weights "
                         "(looked for model.safetensors / pytorch_model.bin)")

    model = CustomTransformerModel(config)
    if weights.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(weights)
    else:
        state = torch.load(weights, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state dict does not match the model built from {cfg_path}\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            "refusing to run on a partially initialised model")
    # eval() disables dropout (encoder 0.1) so the attribution is deterministic
    # and describes the model as it is evaluated. It does NOT disable autograd --
    # that is torch.no_grad(), which this analysis must never use.
    return model.to(device).eval(), config


def build_test_dataset(run, config, verbose=True):
    """The held-out participant's test split, rebuilt exactly as train.py built it.

    observation_time comes from the CHECKPOINT's config; everything else from the
    run's frozen log-derived spec. Anything that changes a tensor's shape must
    come from the checkpoint (trajectory_simple and max_obb_tokens are the two
    that bite), and anything that changes tensor CONTENT must come from the log
    (time_anchor and fusion are the two that bite there).
    """
    spec = runspec.RUNS[run]
    kwargs = dict(runspec.DATASET_KWARGS)
    kwargs.update(
        observation_time=config.observation_time,
        max_obb_tokens=config.max_obb_tokens,
        trajectory_simple=config.trajectory_simple,
        env_class_averaging=config.env_class_averaging,
        env_max_classes_per_voxel=config.env_max_classes_per_voxel,
    )
    ds = HDEpicInteractionDataset('test', participants=list(spec['test_participants']),
                                  verbose=verbose, **kwargs)
    if len(ds) != spec['n_test_episodes']:
        raise SystemExit(
            f"rebuilt test split for {run} has {len(ds)} episodes, the training run "
            f"logged {spec['n_test_episodes']}. The dataset does not match the one the "
            "checkpoint was evaluated on -- refusing to attribute over different data.")
    return ds


def collate(dataset, indices, device):
    batch = torch_default_data_collator([dataset[int(i)] for i in indices])
    return {k: (v.to(device) if torch.is_tensor(v)
                else {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict)
                else v)
            for k, v in batch.items()}


def saved_predictions(run, which):
    """The logits CustomTrainer saved for this checkpoint, [N, 4096] float32.

    indexes.pkl is 0..N-1 in order (the eval dataloader does not shuffle), so
    row i corresponds to dataset[i]. Verified for both runs.
    """
    d = runspec.checkpoint_dir(run, which)
    with open(os.path.join(d, 'predictions.pkl'), 'rb') as f:
        preds = np.asarray(pickle.load(f))
    with open(os.path.join(d, 'indexes.pkl'), 'rb') as f:
        idx = np.asarray(pickle.load(f))
    if not np.array_equal(idx, np.arange(len(idx))):
        raise SystemExit(f"{d}/indexes.pkl is not 0..N-1; row i does not map to dataset[i]")
    return preds


# ---------------------------------------------------------------------------
# token layout
# ---------------------------------------------------------------------------

class TokenLayout:
    """Sequence spans per segment, read off the model rather than hardcoded.

    model.forward builds a ``segment_layout`` list for exactly this purpose and
    returns it as ``outputs['segments']`` under need_attention. Reading it there
    means this analysis cannot drift from the assembly order in model.py:1515,
    which is the failure that would silently mislabel every attribution.
    """

    def __init__(self, spans, seq_len):
        self.spans = list(spans)             # [(name, start, end)], 'sep' repeated
        self.seq_len = int(seq_len)
        # 'sep' appears three times; merge the repeats into one reported group.
        self.groups = {}
        for name, start, end in self.spans:
            self.groups.setdefault(name, []).append((start, end))
        self.names = [n for n in ('cls', 'video', 'sep', 'location', 'obb', 'trajectory')
                      if n in self.groups]

    def index_mask(self, name):
        """[L] bool -- True at the positions belonging to this segment group."""
        m = np.zeros(self.seq_len, dtype=bool)
        for start, end in self.groups[name]:
            m[start:end] = True
        return m

    def n_slots(self, name):
        return int(sum(end - start for start, end in self.groups[name]))

    def as_dict(self):
        return {'seq_len': self.seq_len,
                'spans': [[n, int(s), int(e)] for n, s, e in self.spans],
                'groups': {n: [[int(s), int(e)] for s, e in v] for n, v in self.groups.items()}}

    def describe(self):
        out = [f"sequence length L = {self.seq_len}"]
        for name, start, end in self.spans:
            out.append(f"  [{start:4d}, {end:4d})  {name:12s} n={end - start}")
        return "\n".join(out)


def read_layout(model, dataset, device):
    """One B=1 forward with need_attention to obtain the model's own segment list.

    need_attention materialises [1, 4096, L] (~9.7 MB at L=592), which is why this
    runs once on a single episode and is never used on the analysis path.
    """
    batch = collate(dataset, [0], device)
    with torch.no_grad():
        out = model(**batch, need_attention=True)
    spans = out['segments']
    seq_len = out['attention'].shape[-1]
    if spans[-1][2] != seq_len:
        raise SystemExit(f"segment spans end at {spans[-1][2]} but attention has "
                         f"{seq_len} columns -- the layout does not cover the sequence")
    return TokenLayout(spans, seq_len)


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

class Capture:
    """Captures the encoder input and the padding mask, and can substitute the
    encoder input (which is what Integrated Gradients needs).

    Usage:
        with Capture(model) as cap:
            out = model(**batch)
            g, = torch.autograd.grad(target.sum(), cap.encoder_input, retain_graph=True)

    ``substitute`` is a callable ``x -> x'`` applied to the encoder input before
    the encoder sees it. Returning a tensor that requires grad makes it the
    attribution point; the rest of the forward pass is unchanged, so the padding
    mask is still the real one built from the real batch.
    """

    def __init__(self, model, substitute=None, retain_grad=True, mask_edit=None):
        self.model = model
        self.substitute = substitute
        self.retain_grad = retain_grad
        # mask_edit: [B, L] bool -> [B, L] bool, applied to the padding mask before
        # the encoder sees it and handed on unchanged to the decoder. This is how
        # Method 2 REMOVES a token: the architecture already treats padding as
        # absent (Method 1 measured a padded token's gradient as exactly 0), so
        # masking is genuine removal rather than substitution of some invented
        # vector. The same edited mask must reach both the encoder and the decoder
        # -- handing the decoder the original would leave the "removed" tokens
        # readable by the 4096 queries, which is a silent correctness bug.
        self.mask_edit = mask_edit
        self.encoder_input = None
        self.pad_mask = None
        self._edited_mask = None
        self._handles = []

    def _pre_encoder(self, module, args, kwargs):
        x = args[0]
        if self.substitute is not None:
            x = self.substitute(x)
        if self.retain_grad and x.requires_grad:
            # Non-leaf tensors drop their .grad by default; this keeps it, and
            # autograd.grad(..., inputs=x) works on it either way.
            x.retain_grad()
        self.encoder_input = x
        if self.mask_edit is not None:
            m = kwargs.get('src_key_padding_mask')
            if m is None and len(args) > 2:
                m = args[2]
            if m is not None:
                m = self.mask_edit(m)
                if bool(m.all(dim=1).any()):
                    raise SystemExit(
                        "mask_edit masked every token of some row; attention's softmax "
                        "returns NaN for a fully masked query. Keep at least one token.")
                self._edited_mask = m
                kwargs = dict(kwargs)
                kwargs['src_key_padding_mask'] = m
        return (x,) + tuple(args[1:]), kwargs

    def _pre_decoder(self, module, args, kwargs):
        # decoder(transformer_output, attention_mask, need_weights=...) -- model.py:1605
        if self._edited_mask is not None:
            self.pad_mask = self._edited_mask
            if len(args) > 1:
                return (args[0], self._edited_mask) + tuple(args[2:]), kwargs
            kwargs = dict(kwargs)
            kwargs['memory_key_padding_mask'] = self._edited_mask
            return args, kwargs
        if len(args) > 1:
            self.pad_mask = args[1]
        else:
            self.pad_mask = kwargs.get('memory_key_padding_mask')
        return None

    def __enter__(self):
        self._handles.append(
            self.model.transformer.register_forward_pre_hook(self._pre_encoder, with_kwargs=True))
        if getattr(self.model, 'decoder', None) is not None:
            self._handles.append(
                self.model.decoder.register_forward_pre_hook(self._pre_decoder, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
# All four are per-sample scalars, shape [B]. Samples never interact inside the
# model (attention is within-sample, LayerNorm is per position), so ONE backward
# on target.sum() yields the correct per-sample gradient in each batch column --
# which is what makes batched attribution valid rather than an approximation.

TARGETS = ('pos_contrast', 'pos', 'loss', 'top20')
PRIMARY_TARGET = 'pos_contrast'
TOP_K = 20   # ~ the 20.8 positive voxels per episode measured on the train splits

TARGET_DOC = {
    'pos_contrast': "mean logit over ground-truth positive voxels MINUS the episode's "
                    "mean logit over all 4096. PRIMARY. Removes the per-episode offset, "
                    "so it measures what makes the GT voxels stand out rather than what "
                    "raises the whole map. Chosen because ~91% of logit variance in this "
                    "architecture is a static per-voxel field arriving through the decoder's "
                    "query residual and bias, neither of which depends on input tokens.",
    'pos':          "mean logit over ground-truth positive voxels. The plain 'ground-truth "
                    "target score'. Includes the episode offset.",
    'loss':         "per-sample BCEWithLogits, mean over the 4096 voxels -- the model's own "
                    "training objective. NOTE the sign: higher loss is WORSE, so a positive "
                    "attribution here means the token pushes the loss UP, i.e. hurts.",
    'top20':        f"mean of the {TOP_K} highest logits -- the model's own prediction rather "
                    "than the ground truth. The analogue of 'the predicted-class logit'.",
}


def compute_targets(logits, labels):
    """dict name -> [B] scalars, all differentiable w.r.t. logits."""
    target = labels.float()
    n_pos = target.sum(dim=1)
    if (n_pos <= 0).any():
        # The loader drops empty-GT episodes, so this should never fire. If it
        # does, the split is not the one the run used.
        raise SystemExit("an episode has no positive voxels; empty-GT episodes are "
                         "supposed to be dropped by the loader")
    mean_all = logits.mean(dim=1)
    mean_pos = (logits * target).sum(dim=1) / n_pos
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none').mean(dim=1)
    topk = logits.topk(TOP_K, dim=1).values.mean(dim=1)
    return {'pos_contrast': mean_pos - mean_all,
            'pos': mean_pos,
            'loss': bce,
            'top20': topk}


# ---------------------------------------------------------------------------
# scoring + aggregation
# ---------------------------------------------------------------------------

SCORES = ('grad_l2', 'gxi', 'abs_gxi')

SCORE_DOC = {
    'grad_l2': "||dT/dx_p||_2 over the 768 hidden dims. SENSITIVITY: how much the target "
               "moves for a small nudge to this token. Non-negative, and independent of "
               "what the token actually contains.",
    'gxi':     "sum_d (dT/dx_p)_d * (x_p)_d, i.e. <grad, input>. SIGNED CONTRIBUTION: the "
               "first-order estimate of how much the target would fall if this token were "
               "zeroed. Positive = the token pushes the target up.",
    'abs_gxi': "|gxi|, for rankings and for shares (a signed quantity has no meaningful share).",
}


def token_scores(grad, x):
    """grad, x: [L, B, H] -> dict of [B, L] numpy arrays (batch-first for storage)."""
    gl2 = grad.norm(dim=-1)                 # [L, B]
    gxi = (grad * x).sum(dim=-1)            # [L, B]
    return {'grad_l2': gl2.T.detach().cpu().numpy(),
            'gxi': gxi.T.detach().cpu().numpy(),
            'abs_gxi': gxi.abs().T.detach().cpu().numpy()}


# ---------------------------------------------------------------------------
# channel decomposition
# ---------------------------------------------------------------------------
# Every token in these runs is an EXACT sum of named parts, because
# config.token_layernorm=False makes model._ln the identity:
#
#   video      = visual_mapper(features) + time_code
#   location   = pos_code + location_type + time_code          (location_tokens='joint')
#   obb        = pos_code + class + volume + obb_type
#   trajectory = pos_code + distance + trajectory_type + class (trajectory_token='end_distance')
#
# grad x input is bilinear, so <grad, token> = sum_parts <grad, part> exactly. That
# turns "how much of a modality's contribution flows through the SHARED spatial
# code" into a measurement rather than a caveat -- pos_code is one function
# (model._voxel_pos_embedding, norm = spatial_gain) used by location, obb,
# trajectory AND the decoder's 4096 queries, and class_embedding is shared by obb
# and trajectory. Their scores are correlated by construction; this says how much.
#
# Only the SIGNED gxi score decomposes. A norm is not additive, so grad_l2 cannot
# be split this way.
CHANNELS = ('visual', 'time', 'pos_code', 'class', 'volume', 'distance',
            'type_marker', 'cls_token', 'sep_token')

SHARED_CHANNELS = ('pos_code', 'class')   # parameters serving more than one modality


def token_components(model, batch, layout, encoder_input):
    """dict channel -> [L, B, H], summing exactly to the encoder input.

    Recomputed from the batch with the model's own methods rather than hooked out
    of forward, because forward never materialises the parts separately. The
    caller MUST check the sum against encoder_input (verify_components below):
    that check is what stops this drifting from model.forward if the assembly
    there ever changes.
    """
    cfg = model.config
    if getattr(cfg, 'token_layernorm', False):
        raise SystemExit("channel decomposition assumes token_layernorm=False "
                         "(otherwise a token is a LayerNorm of a sum, not a sum)")
    if getattr(cfg, 'location_tokens', 'axis') != 'joint':
        raise SystemExit("channel decomposition implements location_tokens='joint' only")
    if getattr(cfg, 'trajectory_token', 'mapper') != 'end_distance':
        raise SystemExit("channel decomposition implements trajectory_token='end_distance' only")

    L, B, H = encoder_input.shape
    dev, dt = encoder_input.device, encoder_input.dtype
    ch = {c: torch.zeros(L, B, H, device=dev, dtype=dt) for c in CHANNELS}
    g = layout.groups
    obs_time = batch['visual_features'].size(1)
    time_enc = model._time_encoding(obs_time, dev, dt)          # [T, H] or None

    def marker(vec, s, e):
        return vec.reshape(1, 1, H).expand(e - s, B, H)

    if 'cls' in g:
        s, e = g['cls'][0]
        ch['cls_token'][s:e] = model._ln('cls', model.cls_token).expand(-1, B, -1)
    if 'sep' in g:
        sep = model._ln('sep', model.sep_token)
        for s, e in g['sep']:
            ch['sep_token'][s:e] = marker(sep, s, e)
    if 'video' in g:
        s, e = g['video'][0]
        ch['visual'][s:e] = model._ln(
            'video', model.visual_mapper(batch['visual_features'])).permute(1, 0, 2)
        if time_enc is not None:
            ch['time'][s:e] = time_enc.unsqueeze(1).expand(-1, B, -1)
    if 'location' in g:
        s, e = g['location'][0]
        ch['pos_code'][s:e] = model._voxel_pos_embedding(
            batch['person_locations']).permute(1, 0, 2)
        ch['type_marker'][s:e] = marker(model.location_type_embedding, s, e)
        if time_enc is not None:
            ch['time'][s:e] = time_enc.unsqueeze(1).expand(-1, B, -1)
    if 'obb' in g:
        s, e = g['obb'][0]
        ch['pos_code'][s:e] = model._voxel_pos_embedding(batch['obb_voxels']).permute(1, 0, 2)
        ch['class'][s:e] = model._class_embed(batch['obb_classes']).permute(1, 0, 2)
        ch['volume'][s:e] = model.volume_embedding(batch['obb_volumes']).permute(1, 0, 2)
        ch['type_marker'][s:e] = marker(model.obb_type_embedding, s, e)
    if 'trajectory' in g:
        s, e = g['trajectory'][0]
        ch['pos_code'][s:e] = model._voxel_pos_embedding(
            batch['trajectory_end_voxel']).permute(1, 0, 2)
        ch['distance'][s:e] = model.distance_embedding(
            batch['trajectory_distance']).permute(1, 0, 2)
        ch['type_marker'][s:e] = marker(model.trajectory_type_embedding, s, e)
        if batch.get('trajectory_classes') is not None:
            ch['class'][s:e] = model._class_embed(
                batch['trajectory_classes']).permute(1, 0, 2)
    return ch


def verify_components(ch, encoder_input, tol=1e-2):
    """max |sum(channels) - encoder_input|. Must be float rounding, not a gap.

    A missing channel would show up as a residual of order the missing vector's
    norm (28-48 here), so any tolerance far below that catches it. Returns the
    measured residual so it can be recorded rather than merely asserted.
    """
    total = None
    for v in ch.values():
        total = v if total is None else total + v
    resid = float((total - encoder_input).abs().max())
    if resid > tol:
        raise SystemExit(
            f"channel decomposition does not reconstruct the encoder input "
            f"(max residual {resid:.4g} > {tol}). model.forward's token assembly has "
            "changed and common.token_components must be updated to match.")
    return resid


def valid_matrix(pad_mask):
    """[B, L] bool -- True where a token is real (attendable), from the model's own mask."""
    return (~pad_mask).detach().cpu().numpy()


def aggregate(score_bl, valid_bl, layout):
    """Per-segment totals and valid-token counts.

    Returns (total [B, S], n_valid [B, S]) over layout.names. Padding is excluded
    by multiplication with the validity mask; it must already be zero for a
    gradient score, and asserting that is the caller's job (sanity check 2).
    """
    names = layout.names
    B = score_bl.shape[0]
    total = np.zeros((B, len(names)), dtype=np.float64)
    nval = np.zeros((B, len(names)), dtype=np.int32)
    for s, name in enumerate(names):
        idx = layout.index_mask(name)
        sub_score = score_bl[:, idx]
        sub_valid = valid_bl[:, idx]
        total[:, s] = (sub_score * sub_valid).sum(axis=1)
        nval[:, s] = sub_valid.sum(axis=1)
    return total, nval


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_manifest():
    import platform
    import transformers
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'transformers': transformers.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'hostname': platform.node(),
    }


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
