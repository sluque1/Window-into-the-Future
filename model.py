import math

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from boxer_env import MAX_OBB_TOKENS as _default_max_obb_tokens
from boxer_env import VOLUME_VOCAB_SIZE as _default_volume_vocab_size
from decoder import VoxelQueryDecoder, fourier_code, fourier_scale_per_axis, mlp_hidden_sizes
from object_class_vocab import NUM_OBJECT_CLASSES as _default_num_object_classes
from trajectory_tokens import trajectory_raw_dim as _default_trajectory_raw_dim
from trajectory_tokens import DISTANCE_VOCAB_SIZE as _default_distance_vocab_size

GRID = 16   # voxel-grid resolution per axis, matches loader.py's GRID

# Row order of the unified modality-marker table (config.type_embeddings='unified').
# A marker says "this token arrived via THIS modality", which matters because the
# encoder has no positional encoding over the sequence: without one, an OBB token
# and a joint-location token at the same voxel would be the identical vector.
# 'env' is absent deliberately -- it has never carried a marker, and adding one
# would change every env run for a collision that does not arise (an env token is
# a class embedding at a voxel, never confusable with a per-second stream).
TYPE_EMB_INDEX = {'video': 0, 'location': 1, 'obb': 2, 'gaze': 3, 'trajectory': 4}
N_TYPE_EMB = len(TYPE_EMB_INDEX)


def sigmoid_focal_loss(logits, target, gamma, alpha):
    """Focal loss (Lin et al., ICCV 2017, Eq. 4-5), mean-reduced.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability assigned to the TRUE class. The (1-p_t)^gamma
    factor is the whole idea: an easy negative the model already scores at
    p=0.99 contributes (0.01)^2 = 1e-4 of its usual loss, so the ~4070 empty
    voxels in a 4096-voxel target stop drowning out the ~20 positive ones.

    gamma=0 reduces this to plain (alpha-weighted) BCE exactly, which is what
    makes gamma a clean ablation axis rather than an on/off switch.

    Built on binary_cross_entropy_with_logits rather than an explicit log, so
    the log-sum-exp stabilisation is PyTorch's and not ours.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    loss = bce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
        loss = alpha_t * loss
    return loss.mean()


def resolve_position_encoding(config):
    """'fourier' or 'none' -- what the INPUT tokens' positional code is.

    'auto' (the default) ties the input side to the decoder's query
    parameterisation, so ``--voxel_query_type fourier`` makes queries and input
    tokens speak one language rather than two unrelated ones. Under the default
    ``voxel_query_type='axis_sum'`` it resolves to 'none' and nothing changes.
    getattr, not attribute access: a config.json written before this landed has
    no such key and must load as the pre-change model rather than crash.
    """
    pe = getattr(config, 'position_encoding', 'auto')
    if pe != 'auto':
        return pe
    return 'fourier' if getattr(config, 'voxel_query_type', 'axis_sum') == 'fourier' else 'none'


class CustomConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for the HD-EPIC interaction prediction model.

    PretrainedConfig is just a typed attribute container; it provides save/load utilities
    but does not define any neural-network behaviour.
    """
    model_type = "hd_epic_transformer"

    def __init__(
        self,
        visual_feature_dim=4096,       # Dimensionality of EgoVLPv2 per-frame embeddings
        transformer_hidden_size=768,   # d_model for all transformer layers (must be divisible by num_heads)
        num_transformer_layers=6,      # Number of stacked TransformerEncoderLayer blocks
        num_heads=8,                   # Number of attention heads (768 / 8 = 96 dims per head)
        output_dim=4096,               # Output logit vector size (== number of voxels, 16^3)
        num_object_classes=None,       # Vocabulary size for class_embedding; None = object_class_vocab.py's default
                                        # (0=empty/pad, 1..1220=boxer sem_id+1 (LVIS), 1221=trajectory-only OOV)
        observation_time=30,           # Sequence length in seconds == transformer input token count for visual
        location_vocab_size=17,        # 0=unknown/pad, 1–16=grid cell index per spatial axis
        use_static_env=False,          # Whether env_grid is [B, 4096] (static) or [B, T, 4096] (temporal)
        input_modality='both',         # comma-separated subset of {'video','location','env','trajectory'}
        trajectory_raw_dim=None,       # dim of one raw trajectory-token feature vector; None = trajectory_tokens.py's default
        trajectory_hidden_dim=256,     # hidden width of the trajectory_mapper MLP
        trajectory_simple=False,       # simplified trajectory tokens: times + voxel coords only
                                        # (embedded via the shared location tables), no interior points
        trajectory_token='mapper',     # composition of a trajectory token. 'mapper' (legacy) sums
                                        # trajectory_mapper(raw features) + class + (under
                                        # trajectory_simple) both endpoint positions. 'end_distance'
                                        # drops the MLP entirely for
                                        #   class + pos(end) + distance_bin + trajectory_type
                                        # -- requires trajectory_simple=True. See the __init__ block.
        distance_vocab_size=None,      # 0=pad, 1..16=log-spaced distance bin; None =
                                        # trajectory_tokens.DISTANCE_VOCAB_SIZE
        class_embedding_init=None,     # path to class_embeddings.npz (OWLv2 text vectors) to seed
                                        # class_embedding with, or None for the legacy random table
        class_embedding_trainable=False,  # under class_embedding_init: whether the 1220 semantic rows
                                        # themselves train. False = frozen semantics, only the
                                        # 512->768 projection learns
        env_positional_encoding=False, # add each voxel's (ix,iy,iz) embedding (shared location tables)
                                        # to its env token, so the model can tell neighbouring voxels apart
        env_class_averaging=False,     # average the class embeddings of every object overlapping a
                                        # voxel (see env_max_classes_per_voxel) instead of one winner
        env_max_classes_per_voxel=4,   # K: max classes averaged per voxel when env_class_averaging=True
        max_obb_tokens=None,           # N tokens in the 'obb' segment; None = boxer_env.MAX_OBB_TOKENS
        volume_vocab_size=None,        # 0=pad, 1..16=log-spaced volume bin; None = boxer_env's default
        readout='token0',              # 'token0' (legacy: decode sequence seat 0, a content token)
                                        # or 'cls' (a dedicated learned token prepended to the sequence)
        voxel_bias_init='zero',        # 'zero' (legacy: nn.Linear's default bias init) or 'prior'
                                        # (log(p/(1-p)) of the train split's positive rate, applied
                                        # in-place by train.py once the split is known)
        decoder='linear',              # 'linear' (legacy: Linear(768->4096) on one token) or
                                        # 'voxel_query' (see decoder.VoxelQueryDecoder)
        voxel_query_type='axis_sum',   # how a voxel coordinate becomes a query vector:
                                        # 'axis_sum' (shared location tables) | 'fourier' | 'learned'
        voxel_bias='factorised',       # decoder's per-voxel prior: 'factorised' (bias_x[i]+bias_y[j]
                                        # +bias_z[k], 48 params) | 'full' ([4096], ablation arm) |
                                        # 'mlp' (small Tanh net of (x,y,z), see decoder.py)
        voxel_bias_mlp_hidden='32,32', # hidden widths for voxel_bias='mlp', comma-separated.
                                        # Ignored otherwise.
        voxel_query_fourier_scale=10.0,  # std of W_r's init for voxel_query_type='fourier'.
                                        # One value, or three comma-separated ones for a
                                        # per-axis init (the grids are not cubic -- see
                                        # fourier_scale_per_axis)
        position_encoding='auto',      # positional code on the INPUT tokens that carry a full
                                        # (i,j,k): obb, env, trajectory endpoints. 'auto' follows
                                        # voxel_query_type so query and input share one code,
                                        # 'none' keeps the index tables, 'fourier' forces it on
        position_encoding_mode='replace',  # 'replace' (input position IS the Fourier code, the exact
                                        # structural analogue of the axis_sum arm) or 'add'
                                        # (index tables + Fourier, spec section 2.3's arm)
        encoder_dropout=0.1,           # nn.TransformerEncoderLayer's own default -- what every
                                        # run before this flag silently used
        decoder_dropout=0.0,           # nn.MultiheadAttention's default; the 4096 queries have
                                        # attended with no dropout at all until now
        obb_token_dropout=0.0,         # train-time probability of hiding each OBB token, so the
                                        # model cannot rely on one identifying object inventory
        location_tokens='axis',        # 'axis' (legacy: 3 tokens/second, one per axis, 90 total)
                                        # or 'joint' (1 token/second carrying the whole position
                                        # through _voxel_pos_embedding, 30 total)
        gaze_tokens='axis',            # the same choice for the gaze stream, and the reason
                                        # 'gaze' was outside --position_encoding's scope: under
                                        # 'axis' (legacy) a gaze token holds one SCALAR, so a 3-D
                                        # code has nowhere to go. 'joint' feeds 1 token/second
                                        # carrying the looked-at voxel through
                                        # _voxel_pos_embedding, 90 tokens -> 30, so gaze finally
                                        # shares the one spatial vocabulary with location, obb
                                        # and the decoder's queries
        loss='bce',                    # 'bce' (legacy: plain BCEWithLogitsLoss, nothing done
                                        # about the ~1:200 imbalance) or 'focal' (Lin et al.,
                                        # ICCV 2017). Note the repo already uses that paper's
                                        # bias init via voxel_bias_init='prior'; this adds the
                                        # loss reshaping that init was designed to accompany
        focal_gamma=2.0,               # focusing parameter; 0 reduces focal exactly to BCE
        focal_alpha=0.25,              # positive-class weight. 0.25 is the paper's value and
                                        # DOWN-weights positives -- counterintuitive at 1:200,
                                        # but the (1-p_t)^gamma term already up-weights hard
                                        # positives, so alpha compensates. -1 disables it
        pos_weight=-1.0,               # >0 puts this weight on the positive class in plain BCE.
                                        # The simpler, older fix; the honest control for 'focal'
        decoder_iters=1,               # how many times the ONE cross-attention block is applied,
                                        # weights shared. 1 is the legacy single pass. >1 gives
                                        # the residual an accumulating h to carry, at zero extra
                                        # parameters (see decoder.py)
        voxel_query_untied=False,      # voxel_query_type='fourier' only: give the decoder
                                        # its OWN W_r/gain/marker instead of the model's, so
                                        # tied-vs-untied is a one-variable comparison. Default
                                        # off, creates no parameter, so every existing
                                        # checkpoint still loads
        position_encoding_gain_init=48.0,  # init for the scalar that sets the code's magnitude.
                                        # sqrt(3*768)=48.0 is what the three summed index rows it
                                        # replaces measure, so the arms start matched
        decoder_heads=8,               # attention heads in the decoder (encoder uses num_heads)
        decoder_ffn=False,             # add an FFN sublayer after cross-attention (+4.72M params,
                                        # which makes the model larger than the layer replaced)
        decoder_ffn_dim=3072,          # width of that FFN when enabled
        time_encoding='none',          # 'none' (legacy: the 30 seconds are an unordered set),
                                        # 'fourier' (learned Fourier encoding of time-to-anchor
                                        # added to every per-second token; see docs/
                                        # temporal_spatial_encoding_spec.md), or 'lookup'
                                        # (MMT's T_1..T_D: one learned vector per slot,
                                        # Gabeur et al., ECCV 2020, arXiv:2007.10639)
        time_encoding_scale=10.0,       # std of the time W_r's init -- NOT a range for delta,
                                        # which is always normalised to [0,1). 'fourier' only
        time_encoding_max_wavelength=0.0,  # 'sinusoidal' only: longest wavelength of the
                                        # geometric ladder, in slots. 0 = 2*observation_time
                                        # (the literature's 10000 is dead over a 30-slot window)
        time_encoding_gain='fixed',    # magnitude of the time code. 'fixed' (legacy) divides by
                                        # sqrt(D) for a norm of 0.707, which is 2.6% of a location
                                        # token; 'learned' makes the code unit-norm and multiplies
                                        # by a trainable scalar (time_encoding_gain_init)
        time_encoding_gain_init=8.0,   # that scalar's init. Against LayerNorm'd tokens (norm
                                        # sqrt(768)=27.7) 8.0 is a 29% contribution. Trainer gives it
                                        # its own lr=1e-3 group, so it can travel from here
        token_layernorm=False,         # LayerNorm each segment's tokens before concatenation, so
                                        # video (5.9), location (27.7), obb (66.9) and cls (0.56)
                                        # do not enter the encoder at wildly different magnitudes
        type_embeddings='separate',    # modality markers: 'separate' (legacy -- four independently
                                        # named parameters, and NO marker for video) | 'unified'
                                        # (one [5, H] type_emb table covering video too, added
                                        # AFTER the segment LayerNorm). See TYPE_EMB_INDEX.
        type_embedding_scale=1.0,      # std multiplier on the unified table's randn init. 1.0 gives
                                        # each marker norm sqrt(768)=27.7, i.e. the same magnitude as
                                        # a LayerNorm'd token -- what the legacy markers already are.
        class_embedding_gain=False,    # under class_embedding_init: give the projected class vectors
                                        # a learned magnitude instead of whatever nn.Linear's fan-in
                                        # init happens to produce (measured 0.71, ~40x quieter than
                                        # every other component of an OBB token). See _class_embed.
        class_embedding_gain_init=None,  # that scalar's init; None = sqrt(hidden) = 27.71, matching
                                        # the learned rows and the volume/marker terms beside it
        decoder_residual='raw',        # how the query re-enters after cross-attention:
                                        # 'raw' (legacy: h = q + attn_out, 50:1 in favour of
                                        # the query) | 'normed' (ln_q(q) + attn_out, 25:1)
                                        # | 'balanced' (ln_q(q) + ln_a(attn_out), 1:1)
                                        # | 'evidence' (ln_a(attn_out) -- q is the query and
                                        # nothing else, so it never reaches the output)
                                        # | 'concat' ([ln_q(q); ln_a(attn_out)], head 1536->1,
                                        # so the head weights the two independently)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_feature_dim = visual_feature_dim
        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.num_object_classes = num_object_classes or _default_num_object_classes
        self.observation_time = observation_time
        self.location_vocab_size = location_vocab_size
        self.use_static_env = use_static_env
        self.input_modality = input_modality
        self.trajectory_raw_dim = trajectory_raw_dim or _default_trajectory_raw_dim()
        self.trajectory_hidden_dim = trajectory_hidden_dim
        self.trajectory_simple = trajectory_simple
        if trajectory_token not in ('mapper', 'end_distance'):
            raise ValueError(f"Unknown trajectory_token: {trajectory_token!r} "
                             "(expected 'mapper' or 'end_distance')")
        if trajectory_token == 'end_distance' and not trajectory_simple:
            # 'end_distance' reads trajectory_end_voxel/trajectory_distance, and the
            # loader only emits those on the simplified path. Failing here beats
            # discovering it as a None deref one batch into training.
            raise ValueError("trajectory_token='end_distance' requires "
                             "trajectory_simple=True (the loader emits the endpoint "
                             "voxels and the distance bin only on that path)")
        self.trajectory_token = trajectory_token
        self.distance_vocab_size = distance_vocab_size or _default_distance_vocab_size
        self.class_embedding_init = class_embedding_init
        self.class_embedding_trainable = class_embedding_trainable
        self.env_positional_encoding = env_positional_encoding
        self.env_class_averaging = env_class_averaging
        self.env_max_classes_per_voxel = env_max_classes_per_voxel
        self.max_obb_tokens = max_obb_tokens or _default_max_obb_tokens
        self.volume_vocab_size = volume_vocab_size or _default_volume_vocab_size
        if time_encoding not in ('none', 'fourier', 'sinusoidal', 'lookup'):
            raise ValueError(
                f"Unknown time_encoding: {time_encoding!r} "
                "(expected 'none', 'fourier', 'sinusoidal' or 'lookup')")
        self.time_encoding = time_encoding
        self.time_encoding_scale = time_encoding_scale
        self.time_encoding_max_wavelength = time_encoding_max_wavelength
        if time_encoding_gain not in ('fixed', 'learned'):
            raise ValueError(f"Unknown time_encoding_gain: {time_encoding_gain!r} "
                             "(expected 'fixed' or 'learned')")
        self.time_encoding_gain = time_encoding_gain
        self.time_encoding_gain_init = time_encoding_gain_init
        self.token_layernorm = token_layernorm
        if type_embeddings not in ('separate', 'unified'):
            raise ValueError(f"Unknown type_embeddings: {type_embeddings!r} "
                             "(expected 'separate' or 'unified')")
        self.type_embeddings = type_embeddings
        self.type_embedding_scale = type_embedding_scale
        self.class_embedding_gain = class_embedding_gain
        self.class_embedding_gain_init = class_embedding_gain_init
        if readout not in ('token0', 'cls'):
            raise ValueError(f"Unknown readout: {readout!r} (expected 'token0' or 'cls')")
        self.readout = readout
        if voxel_bias_init not in ('zero', 'prior', 'empirical', 'empirical_factorised'):
            raise ValueError(
                f"Unknown voxel_bias_init: {voxel_bias_init!r} (expected 'zero', 'prior', "
                "'empirical' or 'empirical_factorised')")
        self.voxel_bias_init = voxel_bias_init
        if decoder not in ('linear', 'voxel_query'):
            raise ValueError(
                f"Unknown decoder: {decoder!r} (expected 'linear' or 'voxel_query')")
        self.decoder = decoder
        if voxel_query_type not in ('axis_sum', 'fourier', 'learned'):
            raise ValueError(f"Unknown voxel_query_type: {voxel_query_type!r} "
                             "(expected 'axis_sum', 'fourier' or 'learned')")
        self.voxel_query_type = voxel_query_type
        if voxel_bias not in ('factorised', 'full', 'mlp'):
            raise ValueError(
                f"Unknown voxel_bias: {voxel_bias!r} (expected 'factorised', 'full' or 'mlp')")
        self.voxel_bias = voxel_bias
        self.voxel_bias_mlp_hidden = voxel_bias_mlp_hidden
        mlp_hidden_sizes(voxel_bias_mlp_hidden)   # validate now, not at build time
        if voxel_bias == 'mlp' and voxel_bias_init not in ('zero', 'prior'):
            raise ValueError(
                f"voxel_bias='mlp' has no closed-form fit for a per-voxel field "
                f"(unlike 'factorised'/'full'), so voxel_bias_init={voxel_bias_init!r} "
                "cannot be applied to it. Use 'zero' or 'prior' and let it learn via "
                "--voxel_bias_lr > 0 instead.")
        # Stored exactly as given (a float or a "sx,sy,sz" string) so config.json
        # stays byte-compatible with every run recorded before the per-axis form
        # existed; fourier_scale_per_axis parses it at the point of use.
        self.voxel_query_fourier_scale = voxel_query_fourier_scale
        fourier_scale_per_axis(voxel_query_fourier_scale)   # validate now, not at build time
        if position_encoding not in ('auto', 'none', 'fourier'):
            raise ValueError(f"Unknown position_encoding: {position_encoding!r} "
                             "(expected 'auto', 'none' or 'fourier')")
        self.position_encoding = position_encoding
        if position_encoding_mode not in ('replace', 'add'):
            raise ValueError(f"Unknown position_encoding_mode: {position_encoding_mode!r} "
                             "(expected 'replace' or 'add')")
        self.position_encoding_mode = position_encoding_mode
        self.position_encoding_gain_init = position_encoding_gain_init
        if location_tokens not in ('axis', 'joint'):
            raise ValueError(f"Unknown location_tokens: {location_tokens!r} "
                             "(expected 'axis' or 'joint')")
        self.location_tokens = location_tokens
        if gaze_tokens not in ('axis', 'joint'):
            raise ValueError(f"Unknown gaze_tokens: {gaze_tokens!r} "
                             "(expected 'axis' or 'joint')")
        self.gaze_tokens = gaze_tokens
        for name, v in (('encoder_dropout', encoder_dropout),
                        ('decoder_dropout', decoder_dropout),
                        ('obb_token_dropout', obb_token_dropout)):
            if not 0.0 <= v < 1.0:
                raise ValueError(f'{name} must be in [0, 1), got {v}')
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.obb_token_dropout = obb_token_dropout
        # A Fourier input code against index-table (or private 'learned') queries is
        # exactly the two-vocabulary split this flag exists to remove -- the query
        # and the input token at one voxel would share no parameters at all. Refuse
        # the pair rather than train it and read the result as a fair comparison.
        if (resolve_position_encoding(self) == 'fourier'
                and decoder == 'voxel_query' and voxel_query_type != 'fourier'):
            raise ValueError(
                f"position_encoding resolves to 'fourier' but voxel_query_type is "
                f"{voxel_query_type!r}: the input tokens and the decoder queries would "
                "then carry unrelated positional codes. Use --voxel_query_type fourier, "
                "or --position_encoding none to keep the index tables on the input side.")
        # Untied is only meaningful for the Fourier queries -- axis_sum shares the
        # location tables by design and 'learned' is private already, so the flag
        # would silently do nothing on either. Refuse rather than mislead.
        if voxel_query_untied and voxel_query_type != 'fourier':
            raise ValueError(
                f"voxel_query_untied=True needs voxel_query_type='fourier', got "
                f"{voxel_query_type!r}: axis_sum queries share the location tables "
                "by construction and 'learned' queries are already private, so "
                "there is nothing for this flag to untie.")
        if loss not in ('bce', 'focal'):
            raise ValueError(f"Unknown loss: {loss!r} (expected 'bce' or 'focal')")
        if loss == 'focal' and pos_weight > 0:
            raise ValueError("pos_weight applies to loss='bce'; focal has its own alpha")
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.pos_weight = pos_weight
        if decoder_iters < 1:
            raise ValueError(f"decoder_iters must be >= 1, got {decoder_iters}")
        # Only 'concat' is genuinely incompatible: h doubles in width every pass.
        #
        # This used to refuse everything except 'raw', which was exactly backwards.
        # Measured on an untrained model: under 'raw', |q|=55.5 against |attn_out|=3.8,
        # so h stays pinned to q and the query barely moves between passes --
        # cos(query_i, query_{i-1}) = 0.9977, and after four iterations
        # cos(h_final, q) = 0.965. Each pass asks the same question and gets the same
        # answer, so 'raw' is the one residual under which depth CANNOT do anything.
        # Refinement needs the two sides comparable, which is what 'balanced' is for.
        if decoder_iters > 1 and decoder_residual == 'concat':
            raise ValueError(
                "decoder_iters>1 cannot use decoder_residual='concat': h doubles in "
                "width on every pass. Use 'balanced' (both sides at sqrt(D), so the "
                "query actually moves), or 'raw' as the deliberately-inert control.")
        self.decoder_iters = decoder_iters
        self.voxel_query_untied = voxel_query_untied
        self.decoder_heads = decoder_heads
        self.decoder_ffn = decoder_ffn
        self.decoder_ffn_dim = decoder_ffn_dim
        if decoder_residual not in ('raw', 'normed', 'balanced', 'evidence', 'concat'):
            raise ValueError(f"Unknown decoder_residual: {decoder_residual!r} "
                             "(expected 'raw', 'normed', 'balanced', 'evidence' "
                             "or 'concat')")
        self.decoder_residual = decoder_residual


class CustomTransformerModel(PreTrainedModel):
    """Transformer encoder model for predicting future object-interaction voxels.

    Architecture overview:
      1. Visual features [B, T, 4096] are projected to hidden dim [B, T, H] via a linear layer.
      2. Actor location [B, T, 3] is embedded separately for each spatial axis → three [B, T, H] sequences.
      3. Environment grid [B, 4096] (or [B, T, 4096]) is embedded to [B, 4096, H].
      4. All sequences are concatenated with separator tokens → one long sequence [B, seq_len, H].
      5. A standard TransformerEncoder runs self-attention over the full sequence.
      6. Decoding, one of two ways (config.decoder):
         'linear'      — the representation at the first position (the first visual token
                         under config.readout='token0', or a dedicated learned [CLS] token
                         prepended to the sequence under 'cls') goes through a single
                         Linear(768 -> 4096), i.e. an independent 768-vector per voxel.
         'voxel_query' — 4096 coordinate queries cross-attend to the whole sequence and one
                         shared Linear(768 -> 1) produces every logit (see decoder.py).
      7. BCEWithLogitsLoss against the binary ground-truth voxel mask.
    """
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)

        # Mirrored onto the MODULE, for the reason the self.loss block below spells
        # out: forward() runs on the model, so a flag that lives only on self.config
        # can be silently ignored by a getattr with a default while every log line
        # claims it is on. getattr with the legacy default keeps checkpoints written
        # before these flags existed loadable.
        self.trajectory_token = getattr(config, 'trajectory_token', 'mapper')
        # Read during __init__ itself (every marker below is built under it), so it
        # has to be mirrored here rather than at the marker sites. The legacy default
        # keeps configs written before this flag existed building the old four
        # separate markers, which is what lets their checkpoints still load.
        self.type_embeddings = getattr(config, 'type_embeddings', 'separate')
        self.distance_vocab_size = getattr(config, 'distance_vocab_size', None) \
            or _default_distance_vocab_size

        # --- Input projections ---

        # Visual mapper: projects each EgoVLPv2 frame embedding from the feature space to the
        # transformer hidden space. Input dim: 4096 (EgoVLPv2 output). Output dim: 768 (d_model).
        self.visual_mapper = nn.Linear(config.visual_feature_dim, config.transformer_hidden_size)

        # Class embedding lookup table: maps an integer class index to a dense vector. Shared
        # between env voxels (env_grid) and trajectory tokens (trajectory_classes, see
        # object_class_vocab.py) so both modalities learn one notion of what each object class
        # means. Index convention (see object_class_vocab.py):
        #   0                  → empty voxel / padding trajectory slot (no object)
        #   1..1220            → Boxer/LVIS sem_id + 1
        #   1221               → trajectory-only OOV (a real track whose name didn't match
        #                        the Boxer/LVIS vocabulary; env never emits this index)
        # Index 0 ("empty") gets a learnable embedding like every other class — matches the
        # FIction_orig object_embedding (no padding_idx).
        self.class_embedding = nn.Embedding(
            config.num_object_classes, config.transformer_hidden_size
        )
        # --- Semantic class initialisation (config.class_embedding_init) ---
        # The legacy table above learns 1222 independent 768-vectors from scratch, so
        # "mug" and "cup" are as unrelated at init as "mug" and "refrigerator", and a
        # class the training kitchens never contained is a vector that never moved.
        # Under LOOCV that is exactly the failure mode: the held-out kitchen has
        # objects the other eight did not.
        #
        # class_embeddings.npz holds OWLv2 text vectors (1220 names x 512, L2-normed)
        # for the same LVIS taxonomy, so it supplies that missing geometry. It is the
        # file iou_fusion already uses to decide which detection classes may merge;
        # here it seeds the model's own notion of what an object is.
        #
        # Rows are joined by NAME through data/boxer_class_vocab.json, not by position:
        # build_class_embeddings sorts names alphabetically, so npz row i is NOT sem_id
        # i. The join is case-insensitive (the vocab says 'cd_player', the npz
        # 'CD_player'), which covers all 1082 real classes; without it 11 silently miss.
        #
        # Frozen semantics + a learned 512->768 projection under the default
        # class_embedding_trainable=False: the projection can rotate and rescale the
        # space to suit the task, but cannot pull two classes apart, which is what
        # fine-tuning 1220 rows on six kitchens would do to the rare ones.
        self._class_semantic_rows = 0
        if getattr(config, 'class_embedding_init', None):
            self._init_semantic_classes(config)
        else:
            self.class_semantic = None
            self.class_proj = None
            self.class_gain = None

        # Learnable separator token: a single vector of shape [768] inserted between input segments.
        # Using nn.Parameter makes it part of the model's parameter set and updated by the optimiser.
        self.sep_token = nn.Parameter(torch.randn(config.transformer_hidden_size))

        # --- Transformer encoder ---
        # TransformerEncoderLayer: one block with multi-head self-attention + feedforward sublayers.
        # batch_first=False means it expects input shape [seq_len, batch_size, d_model].
        # dropout defaults to 0.1 -- nn.TransformerEncoderLayer's own default, which
        # every run before this flag existed silently used. Passing it explicitly
        # changes nothing at 0.1 and makes it tunable above that.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_hidden_size, nhead=config.num_heads,
            dropout=getattr(config, 'encoder_dropout', 0.1), batch_first=False
        )
        # Stack num_transformer_layers encoder layers into a single module. The
        # argparse default is 6, but train_sbatch.sh sets NUM_LAYERS=2 and every
        # production LOOCV run uses 2 -- do not read "6" as what actually runs.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)

        # --- Actor location embeddings ---
        # Three separate embedding tables, one per spatial axis (x, y, z).
        # Vocab size = 17: index 0 = unknown/pad, indices 1–16 = grid cell along that axis.
        # Matches FIction_orig main.py lines 159–167 (three independent axis embeddings).
        self.location_embedding_x = nn.Embedding(config.location_vocab_size,
                                                  config.transformer_hidden_size)
        self.location_embedding_y = nn.Embedding(config.location_vocab_size,
                                                  config.transformer_hidden_size)
        self.location_embedding_z = nn.Embedding(config.location_vocab_size,
                                                  config.transformer_hidden_size)

        # --- Trajectory-token mapper ---
        # Maps each raw trajectory-token feature vector (see trajectory_tokens.py:
        # sinusoidal endpoint times, start/end xyz, K interior shape points,
        # is_ongoing flag) into the shared transformer hidden space. A small
        # MLP (rather than a single linear layer) so the model can learn
        # interactions between these heterogeneous raw features -- e.g. long
        # duration + small displacement -- before self-attention sees them.
        #
        # Not built at all under trajectory_token='end_distance', which has no raw
        # feature vector to map -- so a checkpoint from one layout fails loudly if
        # loaded into the other rather than silently keeping a random MLP.
        self.trajectory_mapper = None
        self.distance_embedding = None
        if self.trajectory_token == 'mapper':
            self.trajectory_mapper = nn.Sequential(
                nn.Linear(config.trajectory_raw_dim, config.trajectory_hidden_dim),
                nn.GELU(),
                nn.Linear(config.trajectory_hidden_dim, config.transformer_hidden_size),
            )
        else:
            # --- trajectory_token='end_distance' ---
            # A track becomes exactly four summed terms, each through the table that
            # already owns that kind of information:
            #   WHAT      -- class_embedding, shared with env and obb
            #   WHERE     -- _voxel_pos_embedding of the END voxel: where the object
            #                was when the observation window closed. For a track still
            #                in motion at obs_end that is the constant-speed
            #                interpolated position (trajectory.reconstruct_linear), not
            #                the annotated destination.
            #   HOW FAR   -- log-spaced bin of the distance travelled inside the window
            #                (trajectory_tokens.DISTANCE_BIN_EDGES), measured on the
            #                interpolated positions before voxel snapping, so it keeps
            #                the sub-voxel resolution the ijk index discards.
            #   MARKER    -- trajectory_type_embedding, below.
            # No time term: a movement that ended 2s ago and one that ended 28s ago are
            # the same token. Deliberate, and the first thing to revisit if this arm
            # underperforms.
            self.distance_embedding = nn.Embedding(self.distance_vocab_size,
                                                    config.transformer_hidden_size)
            # Same rationale as gaze_type_embedding and obb_type_embedding: with no
            # positional encoding over the sequence a token IS its vector, and without
            # this marker a trajectory token would be pos(v) + class -- which is an OBB
            # token at v of that class minus its volume term, and identical to a joint
            # location token at v of an object whose class happened to match. The
            # legacy 'mapper' layout never needed one because the MLP output is a
            # private term no other modality can produce.
            if self.type_embeddings == 'separate':
                self.trajectory_type_embedding = nn.Parameter(
                    torch.randn(config.transformer_hidden_size))

        # Fixed (ix, iy, iz) grid coordinate of every flat voxel index 0..4095, in the
        # same 1..16-per-axis convention as location_embedding_x/y/z (0 reserved for
        # pad/unknown). Used by env_positional_encoding to embed "where" a voxel is,
        # via the SAME per-axis tables as the location/trajectory modalities -- so
        # e.g. voxel (4,10,16) and (3,10,16) share two of three axis embeddings.
        ix, iy, iz = np.unravel_index(np.arange(config.output_dim), (GRID, GRID, GRID))
        env_voxel_ijk = np.stack([ix, iy, iz], axis=-1) + 1   # (4096, 3), values 1..16
        self.register_buffer('env_voxel_ijk', torch.as_tensor(env_voxel_ijk, dtype=torch.long))

        # --- Time encoding (docs/temporal_spatial_encoding_spec.md §1) ---
        # Without this the 30 seconds of video / location / gaze are an unordered SET
        # to self-attention: permute them and the output is unchanged, so "what was I
        # looking at two seconds ago" is not a question the model can ask. That is a
        # strange handicap for an anticipation task, and gaze suffers most -- an
        # unordered gaze sequence is close to a bag of voxels.
        #
        # The encoded quantity is time-to-anchor, delta = (T-1-n)/T in [0,1), NOT
        # absolute time in the recording: relative is invariant to where the window
        # sits, and recency is what should matter. Fourier rather than
        # nn.Embedding(30, 768) for the same reason positional encoding beats a
        # lookup -- delta=3s and delta=4s should be near each other, so the model can
        # learn a smooth recency prior instead of 30 unrelated vectors.
        #
        # time_encoding_scale is the std of W_r's INIT, not a range for delta. Too
        # small and all 30 seconds collapse toward one vector (the model can no longer
        # tell them apart); too large and adjacent seconds alias.
        # getattr, not attribute access: a config.json written before this landed has
        # no such key, and such a checkpoint must load as 'none' rather than crash.
        self.time_encoding = getattr(config, 'time_encoding', 'none')
        if self.time_encoding == 'fourier':
            self.time_W_r = nn.Parameter(
                torch.empty(config.transformer_hidden_size // 2, 1))
            nn.init.normal_(self.time_W_r, std=config.time_encoding_scale)
        elif self.time_encoding == 'sinusoidal':
            # Vaswani et al. 2017. NOT the same as 'fourier' above, and the difference
            # is the spectrum: 'fourier' draws frequencies from N(0, scale), which has
            # ONE characteristic scale, so it trades adjacent-second resolution
            # against window-length range as scale moves. This is a fixed GEOMETRIC
            # ladder, so different dimensions resolve different time scales at once.
            # Measured at T=30 by analysis/time_code_similarity.py -- adjacent-slot
            # cosine, then the gap past which every pair is inside the noise floor:
            #
            #   fourier scale=10   g1 +0.946   informative to gap  6
            #   fourier scale=2    g1 +0.998   informative to gap 29 (but resolves
            #                                  nothing finer than gap 18)
            #   sinusoidal 2T      g1 +0.941   informative to gap 22
            #
            # i.e. the same local sharpness as scale=10 with ~3.7x the range, which is
            # a cell no value of `scale` can reach.
            #
            # max_wavelength defaults to 2*observation_time, NOT the literature's
            # 10000. That default is tuned for sequences of thousands and is actively
            # worse here: over 30 slots most of the ladder is constant, the code's
            # cosine never falls below 0.567, and its dynamic range collapses to 0.406
            # against 0.898 at 2T. Fixed, so it costs no parameters -- but like
            # 'fourier' it is bounded cos/sin, so it still needs time_encoding_gain to
            # be audible.
            mw = float(getattr(config, 'time_encoding_max_wavelength', 0.0) or
                       2 * config.observation_time)
            self.time_max_wavelength = mw
            half = config.transformer_hidden_size // 2
            i = torch.arange(half, dtype=torch.float32)
            self.register_buffer(
                'time_inv_freq',
                1.0 / (mw ** (2 * i / config.transformer_hidden_size)),
                persistent=False)
        elif self.time_encoding == 'lookup':
            # MMT's temporal embeddings (arXiv:2007.10639, eq. Omega(v)=F+E+T):
            # D learned vectors, one per time bin, summed into the token exactly as
            # 'fourier' is. MMT bins by ABSOLUTE second of the video; here the bin is
            # the SLOT, i.e. seconds-before-anchor, keeping the relative convention
            # the loader's end-alignment already enforces. Absolute time would need
            # obs_start threaded into forward() and would stop the same recency
            # behaviour transferring between windows at different video offsets.
            #
            # THIS IS THE ARM THAT NEEDS NO GAIN. nn.Embedding inits N(0,1), so a row
            # sits at sqrt(H)=27.7 -- the same scale as a location index row, and 39x
            # the 0.707 that 'fourier' + gain='fixed' produces. The amplitude problem
            # that made the te10g8 runs a null result cannot arise here, because the
            # magnitude is the weights themselves rather than a bounded cos/sin that
            # has to be handed a scale.
            #
            # The cost is the one CLAUDE.md names: 30 unrelated vectors with no prior
            # that slot 5 and slot 6 are adjacent. That trade has never been measured
            # -- analysis/time_code_similarity.py is what measures it.
            self.time_table = nn.Embedding(config.observation_time,
                                           config.transformer_hidden_size)
            # AMPLITUDE-MATCHED INIT, deliberately not MMT's N(0,1). nn.Embedding's
            # default puts a row at sqrt(768)=27.6, and measured against this config's
            # trained tokens (real features, checkpoint-1881) that is 51% of a
            # location/gaze token and 507% of a video token -- while the fourier and
            # sinusoidal arms sit at time_encoding_gain_init=8.0, i.e. 14.6% and 145%.
            # Running them side by side at those magnitudes would confound SPECTRUM
            # with AMPLITUDE, which is precisely the error that made the te10g8 runs
            # uninterpretable, only inverted.
            #
            # It cannot fix itself either: gain='learned' is refused here (the table
            # IS the magnitude) so there is no scalar in trainer.FAST_LR, and a plain
            # weight at lr=1e-5 moves negligibly over the ~1.9k steps of a 1-epoch
            # run. The init is therefore the amplitude for the whole run, so it has to
            # be right at step 0.
            #
            # std = gain_init / sqrt(H) puts the row norm at gain_init, matching the
            # other arms exactly. MMT's literal N(0,1) is still reachable by setting
            # --time_encoding_gain_init to sqrt(768) = 27.71.
            nn.init.normal_(
                self.time_table.weight,
                std=float(config.time_encoding_gain_init) /
                math.sqrt(config.transformer_hidden_size))

        # --- Time-code magnitude (config.time_encoding_gain) ---
        # W_r sets the FREQUENCIES, never the amplitude: cos/sin are bounded, so
        # [cos, sin] always has norm sqrt(D/2) whatever W_r does, and the legacy
        # /sqrt(D) pins the code at 0.707. Measured against the tokens it is added
        # to on a trained te10 checkpoint: 2.6% of a location token (27.7, one axis
        # embedding row) and 12% of a visual token (5.9). A term that small cannot earn a gradient large
        # enough to matter, and W_r has no way to grow -- over 19k steps it moved
        # 0.025% from its init. 'learned' gives the code the magnitude dimension it
        # was missing: unit-norm the vector and multiply by a trainable scalar, so
        # the model chooses how loud recency is instead of us guessing.
        # torch.tensor draws no random numbers, so this consumes no RNG and every
        # other parameter's init is unchanged by the flag.
        self.time_gain = None
        if getattr(config, 'time_encoding_gain', 'fixed') == 'learned':
            if self.time_encoding not in ('fourier', 'sinusoidal'):
                # Both of those are bounded cos/sin and so cannot set their own
                # amplitude. 'lookup' is excluded deliberately, not by oversight: its
                # magnitude IS its weights, so a separate scalar would be a redundant
                # reparameterisation of something the table can already learn.
                raise ValueError("time_encoding_gain='learned' needs "
                                 "time_encoding='fourier' or 'sinusoidal'; 'none' has "
                                 "no code to scale and 'lookup' sets its own magnitude")
            self.time_gain = nn.Parameter(
                torch.tensor(float(config.time_encoding_gain_init)))

        # --- Per-segment input LayerNorm (config.token_layernorm) ---
        # Each segment's scale is an accident of which init convention built it.
        # nn.Embedding is N(0,1), so one row has norm sqrt(768)=27.7 -- which is a
        # location or gaze token, since those are fed as three separate one-row
        # streams -- and a token summing five rows (obb) lands at 66.9.
        # nn.Linear is U(+-1/sqrt(fan_in)) with fan_in=4096, so visual_mapper's
        # output is 5.9 -- an order of magnitude quieter than the segments it
        # competes with for attention. cls_token (std 0.02) is quieter still at
        # 0.56, and it is the readout seat. nn.TransformerEncoderLayer is post-LN
        # by default, so layer 1's attention sees exactly these raw magnitudes.
        # One LayerNorm per segment TYPE (not per token) puts them all at sqrt(D)
        # while leaving each segment free to learn its own gamma/beta.
        # nn.LayerNorm's init draws no random numbers either, so this flag also
        # leaves every other parameter's init untouched.
        self.token_ln = None
        if getattr(config, 'token_layernorm', False):
            self.token_ln = nn.ModuleDict({
                name: nn.LayerNorm(config.transformer_hidden_size)
                for name in ('video', 'location', 'env', 'obb', 'gaze',
                             'trajectory', 'sep', 'cls')})

        # --- Gaze tokens ---
        # Gaze says "at second i the actor was looking at voxel (ix,iy,iz)", and is
        # fed in exactly like person_locations: split per axis into three 30-token
        # sequences, each embedded through the SAME location_embedding_x/y/z table,
        # so one voxel index means one learned thing whether it arrived via gaze,
        # the actor's own position, a trajectory endpoint or an env voxel. This
        # marker is added on top to say "and this one arrived via gaze" -- required
        # because the encoder has no positional encoding over the sequence, so
        # without it a gaze token and a location token holding the same index would
        # be byte-identical and self-attention could not separate them.
        if self.type_embeddings == 'separate':
            self.gaze_type_embedding = nn.Parameter(torch.randn(config.transformer_hidden_size))

        # --- OBB tokens ---
        # The per-box alternative to the env voxel grid: instead of 4096 tokens
        # (one per voxel, most of them empty), one token per fused 3D box --
        # ~162 on average with fixtures (~124 without), 250 at the cap (see
        # boxer_env.MAX_OBB_TOKENS; the old ~240/512 was 'grid' fusion). A box
        # carries exactly three things, and each is embedded through the table
        # that already owns that kind of information:
        #   WHERE  -- the grid index of its centre, via location_embedding_x/y/z,
        #             so a position means one learned thing whether it arrived
        #             via a box, the actor's own location, gaze or a trajectory.
        #   WHAT   -- Boxer sem_id+1, via the SAME class_embedding the env grid
        #             and trajectory tokens use, so "what a kettle is" is learned
        #             once across every modality that mentions one.
        #   HOW BIG-- a log-spaced volume bin (boxer_env.VOLUME_BIN_EDGES); this
        #             is the one genuinely new signal, since the voxel grid could
        #             only express size as "how many voxels did it fill".
        # Summed, not concatenated, to stay in the shared 768-dim hidden space.
        self.volume_embedding = nn.Embedding(config.volume_vocab_size,
                                             config.transformer_hidden_size)
        # Same rationale as gaze_type_embedding: with no positional encoding over
        # the sequence, an OBB token and an env token at the same voxel holding
        # the same class would otherwise be near-identical vectors.
        if self.type_embeddings == 'separate':
            self.obb_type_embedding = nn.Parameter(torch.randn(config.transformer_hidden_size))

        # --- Output decoder ---
        # 'linear' (legacy): one aggregated [B, 768] vector -> [B, 4096] logits, i.e.
        # an independent 768-vector per voxel. 'voxel_query': 4096 coordinate queries
        # cross-attend to the encoder output and ONE shared head produces each logit
        # (see decoder.py). The two are exclusive, and the unused one is not built --
        # so a pre-change checkpoint loaded into a voxel_query model fails loudly on
        # the missing/unexpected keys instead of silently keeping a random decoder.
        self.decoder = None
        self.output_layer = None
        if config.decoder == 'linear':
            self.output_layer = nn.Linear(config.transformer_hidden_size, config.output_dim)

        # --- Readout token ---
        # 'token0' (legacy) decodes sequence seat 0, which is always the first
        # visual token -- a content token carrying second 0 of the observation
        # window, asked to double as the global summary. 'cls' prepends a
        # dedicated learned token instead, so the summary has a seat of its own.
        # Created AFTER output_layer on purpose: every pre-existing parameter is
        # then drawn from the same RNG stream under both settings, so a
        # --readout cls run differs from its token0 twin by this token alone and
        # not by a wholesale re-draw of the weights.
        self.readout = config.readout
        if self.readout == 'cls':
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, config.transformer_hidden_size))
            nn.init.normal_(self.cls_token, std=0.02)

        # --- Shared spatial Fourier code (config.position_encoding) ---
        # The index tables above are a 48-vector vocabulary for 4096 voxels: e_x[i]
        # + e_y[j] + e_z[k]. Two consequences. They are DISCRETE -- e_x[4] and e_x[5]
        # are independent 768-vectors with no prior that they are adjacent, so
        # neighbouring voxels start out as unrelated as distant ones. And they are
        # ADDITIVE -- every voxel's code decomposes per axis, so no cell can be made
        # distinctive without smearing the same signal across its whole row, column
        # and layer. A Fourier code of the normalised centre fixes both at once:
        # continuous in the coordinate, and W_r mixes all three axes BEFORE cos/sin,
        # so cos(w1*x + w2*y + w3*z) is a genuinely joint function of position.
        #
        # Built here -- after every pre-existing parameter, before the decoder that
        # borrows it -- so that under the default position_encoding='none' nothing
        # is drawn and a default run stays weight-identical to the pre-change model.
        self.obb_token_dropout = float(getattr(config, 'obb_token_dropout', 0.0))
        self.position_encoding = resolve_position_encoding(config)
        self.position_encoding_mode = getattr(config, 'position_encoding_mode', 'replace')
        self.spatial_gain = None
        if self.position_encoding == 'fourier':
            sx, sy, sz = fourier_scale_per_axis(config.voxel_query_fourier_scale)
            self.spatial_W_r = nn.Parameter(
                torch.empty(config.transformer_hidden_size // 2, 3))
            for axis, std in enumerate((sx, sy, sz)):
                nn.init.normal_(self.spatial_W_r[:, axis], std=std)
            # Magnitude, learned -- the lesson of the te10 runs (see the time_gain
            # block above). cos/sin are bounded, so W_r sets frequencies and can
            # never set amplitude; the decoder's legacy /sqrt(D) pinned the code at
            # 0.707, which against a 27.7-norm class embedding is 2.5% of an OBB
            # token and against |attn_out|~1.11 turns decoder_residual='raw' from a
            # 50:1 regime into a 0.6:1 one. Init at sqrt(3*H)=48.0, the measured norm
            # of the three summed index rows this replaces, so the fourier and
            # axis_sum arms start at the same magnitude and differ only in code.
            self.spatial_gain = nn.Parameter(
                torch.tensor(float(config.position_encoding_gain_init)))
            # Padding slots (index 0 on every axis) have no coordinate. The tables
            # gave them row 0, a learned "unknown"; (0-0.5)/16 = -0.031 would instead
            # place them a third of a cell outside the floor corner, i.e. almost
            # exactly on top of voxel 1. Most pad slots are masked out of attention,
            # but not all -- model.forward keeps slot 0 attendable on all-pad OBB
            # rows, where it must read as "no objects here" and not "a box on the floor".
            self.spatial_pad_embedding = nn.Parameter(
                torch.randn(config.transformer_hidden_size))

        # --- Joint location tokens (config.location_tokens='joint') ---
        # Under 'axis' a location token is one axis embedding and nothing else needs
        # saying: the three streams come from three different tables, so an x token
        # and a z token are already distinct. Under 'joint' the token becomes
        # _voxel_pos_embedding(position), which is byte-identical to the positional
        # part of an OBB token and to a decoder query at that voxel -- exactly the
        # collision gaze_type_embedding and obb_type_embedding exist to prevent. So
        # 'joint' needs the marker 'axis' never did. The parameter itself is created
        # AFTER the decoder, below -- see there.
        self.location_tokens = getattr(config, 'location_tokens', 'axis')
        # Same story for gaze, and the same getattr for the same reason: a
        # config.json written before gaze_tokens existed has no such key and must
        # load as 'axis' rather than crash. gaze_type_embedding already exists and
        # is required under BOTH layouts -- under 'joint' more sharply than under
        # 'axis', since a bare joint gaze token is byte-identical to a joint
        # LOCATION token at the same voxel.
        self.gaze_tokens = getattr(config, 'gaze_tokens', 'axis')

        if config.decoder == 'voxel_query':
            # Built last so the encoder-side parameters are drawn from the same RNG
            # stream regardless of which decoder is in use. The location tables are
            # passed in rather than duplicated: a voxel index means one learned
            # thing whether it arrives as an actor position, a gaze point, an OBB
            # centre or -- now -- the place being asked about. Under
            # position_encoding='fourier' the same argument applies to the Fourier
            # code, so the decoder is handed _voxel_pos_embedding itself rather than
            # a copy of its parameters: a query and an input token at one voxel then
            # carry the identical vector by construction, not by two call sites
            # agreeing on a formula.
            self.decoder = VoxelQueryDecoder(
                config, (self.location_embedding_x,
                         self.location_embedding_y,
                         self.location_embedding_z),
                pos_embedding=(self._voxel_pos_embedding
                               if (self.position_encoding == 'fourier'
                                   and not getattr(config, 'voxel_query_untied', False))
                               else None))

        # Marker for the joint location token, created LAST for the same reason
        # cls_token is created after output_layer: every other parameter -- the
        # encoder, the tables, the spatial code, the whole decoder -- is then drawn
        # from an identical RNG stream whichever layout is in use, so a
        # --location_tokens joint run differs from its 'axis' twin by this one
        # appended vector rather than by a wholesale re-draw of the weights. That is
        # what makes the two arms a paired comparison instead of two random inits.
        if self.location_tokens == 'joint' and self.type_embeddings == 'separate':
            self.location_type_embedding = nn.Parameter(
                torch.randn(config.transformer_hidden_size))

        # --- Unified modality-marker table (config.type_embeddings='unified') ---
        # One [5, H] parameter replacing the four named markers above, and adding the
        # one modality that never had one: video. Video cannot COLLIDE with anything
        # (it is the only segment not built from the shared embedding tables), so its
        # marker is about giving every segment the same standing, not disambiguation.
        #
        # Built last, like location_type_embedding and cls_token, so that under the
        # default 'separate' nothing here is drawn and every existing arm keeps its
        # weights bit-for-bit. Note the converse does NOT hold: 'unified' skips the
        # four draws above, so it shifts the RNG stream and is therefore not
        # weight-paired with a 'separate' run -- it is a new arm, not an ablation.
        #
        # Scale matters more than it looks. These markers are added AFTER the segment
        # LayerNorm (see forward), so their norm is an absolute magnitude against a
        # sqrt(H)=27.7 token rather than a share of it. randn (scale 1.0) puts them at
        # exactly 27.7, matching the legacy markers; measured, that takes video's
        # pairwise token cosine from 0.509 to ~0.75. Setting the scale to 0.02 would
        # put them at 0.55, i.e. 2% of the token -- the same inaudibility that made
        # the projected class embedding useless.
        self.type_emb = None
        if self.type_embeddings == 'unified':
            self.type_emb = nn.Parameter(
                torch.randn(N_TYPE_EMB, config.transformer_hidden_size)
                * float(config.type_embedding_scale))

        # Loss function: Binary Cross Entropy with Logits — numerically stable, works on raw logits
        # Applied element-wise over all 4096 voxels, then averaged
        # 'bce' with pos_weight<=0 is the exact legacy object, so the default path is
        # unchanged. torch.tensor draws no RNG, so no other parameter's init moves.
        # Mirrored onto the MODULE, not read off self.config at call time: forward()
        # runs on the model, and `getattr(self, 'loss', 'bce')` against a model that
        # only had config.loss silently selected BCE while every flag and log line
        # said 'focal'. That failure is invisible -- it looks like "focal changed
        # nothing" -- so the attributes live here where forward() actually looks.
        self.loss = getattr(config, 'loss', 'bce')
        self.focal_gamma = float(getattr(config, 'focal_gamma', 2.0))
        self.focal_alpha = float(getattr(config, 'focal_alpha', 0.25))
        pw = getattr(config, 'pos_weight', -1.0)
        self.loss_func = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pw)) if pw > 0 else None)

        # Tells HuggingFace Trainer which key in the batch dict is the "main input" for the model.
        # Here we use 'index' (the sample index scalar) as the nominal input so the Trainer does not
        # try to build attention_mask from it.
        self.main_input_name = "index"

    def apply_voxel_bias_init(self, value):
        """Initialise the output bias (Stage B).

        Called by train.py once the training split is known, since the rate
        depends on future_time AND on which participants are in the split (voxel
        size differs per participant). Applied in place rather than in __init__
        so the model can be constructed before the datasets are built, which
        keeps the RNG consumption order -- and therefore the default path's
        weights -- exactly as they were before this existed.

        A no-op under config.voxel_bias_init='zero' (the default), where the bias
        keeps nn.Linear's own init.

        ``value`` is either a SCALAR (config.voxel_bias_init='prior' -- the class
        prior applied uniformly, carrying no spatial information) or a [4096]
        field in the label flattening order (the 'empirical' arms).

        ⚠️ A per-voxel empirical field IS the training kitchens' spatial layout,
        which under participant-level splits is the thing that must not transfer.
        That is why it is not the default and never should be: the arms exist to
        MEASURE what handing the model that layout does to input-driven learning,
        which is a different purpose from using it. Read any run that sets one as
        an upper bound contaminated by layout, not as a result.

        Sets self.voxel_bias_fit_r2 to the share of the field the parameterisation
        could actually represent (always 1.0 for a scalar or for 'full').

        Returns True if the init was applied.
        """
        if self.config.voxel_bias_init == 'zero':
            return False
        self.voxel_bias_fit_r2 = 1.0
        is_scalar = np.isscalar(value) or getattr(value, 'ndim', 0) == 0
        if self.decoder is not None:
            if is_scalar:
                self.decoder.set_prior_logit(float(value))
            else:
                self.voxel_bias_fit_r2, _, _ = self.decoder.set_voxel_bias(value)
        else:
            # The linear readout's bias is [4096] already, so any field fits it
            # verbatim -- there is no factorised arm on this path.
            with torch.no_grad():
                if is_scalar:
                    self.output_layer.bias.fill_(float(value))
                else:
                    self.output_layer.bias.copy_(
                        torch.as_tensor(value, dtype=self.output_layer.bias.dtype).flatten())
        return True

    def voxel_bias_parameters(self):
        """The bias parameter tensors, whichever readout is in use.

        One place that knows the names, so freezing and the optimizer group
        cannot drift apart from each other or from the decoder's own layout.
        """
        if self.decoder is not None:
            if self.config.voxel_bias == 'factorised':
                return [('decoder.bias_x', self.decoder.bias_x),
                        ('decoder.bias_y', self.decoder.bias_y),
                        ('decoder.bias_z', self.decoder.bias_z)]
            if self.config.voxel_bias == 'mlp':
                return [(f'decoder.bias_mlp.{n}', p)
                        for n, p in self.decoder.bias_mlp.named_parameters()]
            return [('decoder.voxel_bias', self.decoder.voxel_bias)]
        return [('output_layer.bias', self.output_layer.bias)]

    def set_voxel_bias_trainable(self, trainable):
        """Freeze or unfreeze the voxel bias. Returns how many tensors changed.

        Freezing matters more than it looks: measured over 27 finished
        voxel_query runs the bias drifts at most 0.036 from its init at lr=1e-5,
        so it is already frozen in all but name. Making that explicit is what
        lets an init be tested as an init rather than as a starting guess.
        """
        n = 0
        for _, p in self.voxel_bias_parameters():
            p.requires_grad_(bool(trainable))
            n += 1
        return n

    def _init_semantic_classes(self, config):
        """Seed class_embedding from OWLv2 text vectors (see the __init__ block).

        Builds two things: a [V, 512] table of unit text vectors aligned to THIS
        model's class index convention, and the 512 -> 768 projection that maps it
        into the hidden space. Rows with no text vector -- index 0 (empty/pad), the
        OOV row, and the 138 sem_id slots the taxonomy never assigns -- keep the
        learned nn.Embedding rows instead, so "empty voxel" stays a thing the model
        can learn rather than a projection of zeros.
        """
        import json
        import os

        z = np.load(config.class_embedding_init, allow_pickle=False)
        names, emb = [str(n) for n in z['names']], np.asarray(z['emb'], dtype=np.float32)
        # Case-insensitive: the vocab says 'cd_player', the npz 'CD_player'. Without
        # this, 11 of 1082 classes silently fall back to a random row.
        row_of = {n.lower(): i for i, n in enumerate(names)}

        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', 'boxer_class_vocab.json')
        with open(vocab_path) as f:
            sem_to_name = json.load(f)

        vocab_size, dim = config.num_object_classes, emb.shape[1]
        table = np.zeros((vocab_size, dim), dtype=np.float32)
        has_semantic = np.zeros((vocab_size,), dtype=bool)
        missing = []
        for sem_id_str, name in sem_to_name.items():
            row = row_of.get(str(name).lower())
            if row is None:
                missing.append(name)
                continue
            idx = int(sem_id_str) + 1        # class_embedding convention: sem_id + 1
            if idx < vocab_size:
                table[idx] = emb[row]
                has_semantic[idx] = True
        if missing:
            # Loudly, not as a warning: a partial join degrades exactly the rare
            # classes this initialisation exists to help, and is invisible in the
            # metrics. 1082/1082 join against the shipped class_embeddings.npz.
            raise ValueError(
                f"class_embedding_init: {len(missing)} of {len(sem_to_name)} vocabulary "
                f"classes have no vector in {config.class_embedding_init} "
                f"(e.g. {missing[:5]}). Regenerate it with "
                "iou_fusion.build_class_embeddings over data/boxer_class_vocab.json.")

        semantic = torch.from_numpy(table)
        if getattr(config, 'class_embedding_trainable', False):
            self.class_semantic = nn.Parameter(semantic)
        else:
            # A buffer, not a frozen Parameter: it then travels with state_dict and
            # .to(device) but never appears in model.parameters(), so no optimiser
            # group can pick it up by accident.
            self.register_buffer('class_semantic', semantic)
        self.class_proj = nn.Linear(dim, config.transformer_hidden_size, bias=False)
        self.register_buffer('class_has_semantic', torch.from_numpy(has_semantic))
        self._class_semantic_rows = int(has_semantic.sum())

        # --- Magnitude of the semantic class code (config.class_embedding_gain) ---
        # The OWLv2 vectors arrive L2-normed, and class_proj is an nn.Linear with the
        # default fan_in=512 init, so the projected rows land at norm ~0.71 -- against
        # 47.8 (position), 27.7 (volume) and 27.5 (marker) in the same OBB token.
        # Measured on the 2026-08-31 fixt250 runs: 0.708 at init, 0.711 after a full
        # epoch, i.e. lr 1e-5 does not move it. Two class vectors then differ by at
        # most 1.44 against a token norm of 61.7, so swapping an OBB's class rotates
        # its token by under 1.3 degrees -- cup and mug ARE close (cosine 0.765), that
        # closeness is simply inaudible.
        #
        # F.normalize in _class_embed strips amplitude from class_proj entirely and
        # hands it to this scalar, which is what makes the gain a real constraint
        # rather than a redundant reparameterisation of a bias-free Linear (the
        # objection that rules out time_encoding_gain='learned' for 'lookup'). It does
        # NOT move classes relative to one another -- only how loud they all are.
        # Init at sqrt(H) = 27.71, where the learned rows and the volume/marker terms
        # already sit. trainer.FAST_PARAMS gives it lr=1e-3, for the same reason
        # spatial_gain and time_gain need it: a lone scalar never travels at 1e-5.
        self.class_gain = None
        if getattr(config, 'class_embedding_gain', False):
            init = config.class_embedding_gain_init
            if init is None:
                init = math.sqrt(config.transformer_hidden_size)
            self.class_gain = nn.Parameter(torch.tensor(float(init)))

    def _class_embed(self, idx):
        """class_embedding lookup, honouring config.class_embedding_init.

        The one accessor every modality goes through (env, obb, trajectory), so
        "what a kettle is" stays one vector across all of them however it is built.
        Under the legacy path this is exactly nn.Embedding.__call__.
        """
        if self.class_semantic is None:
            return self.class_embedding(idx)
        # Project the whole vocabulary once (1222x512x768, one matmul) rather than
        # per index: env alone would gather 4096 rows per sample, which is strictly
        # more work than building the table.
        projected = self.class_proj(self.class_semantic)
        if self.class_gain is not None:
            # Normalise THEN scale, so class_gain is the only thing setting the code's
            # magnitude -- see the class_gain block in _init_semantic_classes. Rows with
            # no text vector are all-zero here and F.normalize's eps leaves them zero;
            # torch.where discards them anyway in favour of the learned row.
            projected = nn.functional.normalize(projected, dim=-1) * self.class_gain
        table = torch.where(self.class_has_semantic.unsqueeze(-1),
                            projected, self.class_embedding.weight)
        return nn.functional.embedding(idx, table)

    def _time_encoding(self, obs_time, device, dtype):
        """[T, H] Fourier encoding of time-to-anchor, or None when disabled.

        Slot n holds delta = (T-1-n)/T, so the LAST slot is delta=0 (the second
        nearest the anchor) and the first is the oldest. That orientation is not a
        convention chosen here -- it is forced by the loader, which end-aligns all
        three per-second modalities:

          * person_locations / gaze_voxels: loader.py walks t from obs_end-1
            downwards, filling slot T-1 first and leaving gaps as padding.
          * visual_features: padded at the FRONT, with the available features packed
            contiguously -- equivalent to end-alignment only because the EgoVLPv2
            features are dense at 1 Hz (measured: 0 of 3516 consecutive frame
            indices on P08's videos differ from +1s). forward() asserts the shape
            the caller supplies still matches observation_time rather than trusting
            it silently.

        Returns None under config.time_encoding='none', which is what keeps the
        default path bit-identical to the pre-change model.
        """
        if self.time_encoding == 'lookup':
            # Row n is slot n, so the orientation matches the fourier branch below:
            # the LAST row is the second nearest the anchor. The assert is the same
            # contract forward() enforces on the window length -- a table built for
            # 30 slots silently relabelling a 20-slot window would be a correctness
            # bug that trains without error.
            if obs_time != self.time_table.num_embeddings:
                raise ValueError(
                    f"time_encoding='lookup' built {self.time_table.num_embeddings} "
                    f"slots but the window has {obs_time}; observation_time changed "
                    "after the config was written")
            return self.time_table.weight.to(device=device, dtype=dtype)
        if self.time_encoding == 'sinusoidal':
            # Position is the SLOT, so row T-1 is the second nearest the anchor -- the
            # same orientation the fourier branch gets from delta, and the same one
            # the loader's end-alignment guarantees. Vaswani indexes absolute
            # sequence position; here that would be time-since-window-start, which is
            # the same thing up to a constant offset within a fixed-length window.
            pos = torch.arange(obs_time, device=device, dtype=torch.float32).unsqueeze(-1)
            ang = pos * self.time_inv_freq.to(device)                 # [T, H/2]
            te = torch.cat([ang.sin(), ang.cos()], dim=-1)            # [T, H]
            if self.time_gain is None:
                return (te / math.sqrt(self.config.transformer_hidden_size)).to(dtype)
            te = te / math.sqrt(self.config.transformer_hidden_size / 2)
            return (te * self.time_gain).to(dtype)
        if self.time_encoding != 'fourier':
            return None
        delta = (obs_time - 1 - torch.arange(obs_time, device=device, dtype=torch.float32))
        delta = (delta / obs_time).unsqueeze(-1)                  # [T, 1] in [0,1)
        proj = delta @ self.time_W_r.T.float()                    # [T, H/2]
        te = torch.cat([proj.cos(), proj.sin()], dim=-1)          # [T, H]
        if self.time_gain is None:
            # Legacy: norm 0.707, i.e. 2.6% of a location token. See the
            # time_gain block in __init__ for why that is effectively silence.
            return (te / math.sqrt(self.config.transformer_hidden_size)).to(dtype)
        # cos^2 + sin^2 = 1 for each of the H/2 frequencies, so the row norm is
        # exactly sqrt(H/2) and this division makes it exactly 1. The gain is then
        # the code's magnitude in the same units as the token it is added to.
        te = te / math.sqrt(self.config.transformer_hidden_size / 2)
        return (te * self.time_gain).to(dtype)

    def _type(self, name):
        """The modality marker for a segment, or None if this build has none.

        One accessor so the forward pass reads the same under either layout:
        'unified' returns a row of type_emb, 'separate' returns the individually
        named parameter (and None for video, which never had one).
        """
        if self.type_emb is not None:
            return self.type_emb[TYPE_EMB_INDEX[name]]
        return getattr(self, f'{name}_type_embedding', None)

    def _seg(self, name, x):
        """A segment's tokens: LayerNorm the content and attach the modality marker.

        Placement depends on the layout, and deliberately so:

        * 'unified' -- marker added AFTER the norm, so its norm is an absolute
          magnitude against a sqrt(D) token rather than a share of it, exactly as
          the time code is applied. This is the requested design.
        * 'separate' (legacy) -- marker added INSIDE, where it has always been.
          That keeps every pre-existing arm bit-identical, down to the float32
          association order of the sum, which smoke_decoder test 1 pins to 0 and
          which `(a+b+c) + m` vs `(a+b+c+m)` would otherwise break by ~5e-7.
          It also keeps a segment's post-norm norm at sqrt(D) rather than
          sqrt(2)*sqrt(D), which smoke_token_scale asserts.
        """
        marker = self._type(name)
        if marker is None:
            return self._ln(name, x)
        if self.type_embeddings == 'separate':
            return self._ln(name, x + marker)
        return self._ln(name, x) + marker

    def _ln(self, name, x):
        """Per-segment input LayerNorm, or the identity under the default
        config.token_layernorm=False. Applied to a segment's BASE token, before
        the time code is added, so time_gain keeps its meaning as an absolute
        magnitude against a sqrt(D)-normalised token."""
        return x if self.token_ln is None else self.token_ln[name](x)

    def _spatial_encoding(self, ijk):
        """[..., H] Fourier code of a voxel's normalised centre, at unit norm
        times ``spatial_gain``.

        ijk is 1..16 with 0 = pad, so ``(ijk - 0.5)/GRID`` gives a real cell
        exactly the normalised centre ``decoder.voxel_coords`` holds for it,
        ``(ijk0 + 0.5)/GRID``. Those two expressions must stay equal: it is what
        makes a decoder query and an input token at one voxel carry the same code,
        and nothing else in the code enforces it.
        """
        # decoder.fourier_code is the ONE implementation, shared with the decoder's
        # untied queries: cos^2+sin^2 = 1 per frequency makes the raw row norm
        # sqrt(H/2) whatever W_r holds, so dividing by it and multiplying by the
        # gain gives a vector of norm exactly spatial_gain.
        code = fourier_code(ijk, self.spatial_W_r, self.spatial_gain, GRID)
        # Pad slots carry no coordinate -- see the spatial_pad_embedding note in
        # __init__ for why they cannot simply be encoded at (0-0.5)/16.
        is_pad = (ijk == 0).any(dim=-1, keepdim=True)
        return torch.where(is_pad, self.spatial_pad_embedding.to(code.dtype), code)

    def _voxel_pos_embedding(self, ijk):
        """Positional code for a batch of full (i,j,k) grid coordinates, shared by
        every token that carries one: OBB centres, env voxels, simplified
        trajectory endpoints, and -- under config.decoder='voxel_query' -- the
        decoder's queries. One vocabulary, so a voxel means the same learned thing
        wherever it arrives from.

        Under config.position_encoding='none' (the default) this is the sum of the
        three per-axis index tables, also shared with the actor-location and gaze
        modalities. Under 'fourier' it is the continuous code instead ('replace')
        or in addition ('add'). 'add' keeps what a lookup can express and a Fourier
        basis cannot -- non-smooth structure such as "the counter runs along the
        k=8 slab" -- at the cost of no longer being what the queries alone carry.

        ijk: LongTensor [..., 3], 1..16 per axis (0 = pad/unknown, which the index
            tables map to row 0 like everywhere else location is used).
        Returns: FloatTensor [..., transformer_hidden_size].
        """
        if self.position_encoding != 'fourier':
            return (self.location_embedding_x(ijk[..., 0])
                    + self.location_embedding_y(ijk[..., 1])
                    + self.location_embedding_z(ijk[..., 2]))
        code = self._spatial_encoding(ijk)
        if self.position_encoding_mode == 'add':
            code = (code
                    + self.location_embedding_x(ijk[..., 0])
                    + self.location_embedding_y(ijk[..., 1])
                    + self.location_embedding_z(ijk[..., 2]))
        return code

    def forward(self, index, visual_features, visual_mask,
                env_grid=None,
                person_locations=None, person_location_mask=None,
                trajectory_tokens=None, trajectory_mask=None, trajectory_classes=None,
                trajectory_start_voxel=None, trajectory_end_voxel=None,
                trajectory_distance=None,
                env_grid_mask=None,
                gaze_voxels=None, gaze_mask=None,
                obb_voxels=None, obb_volumes=None, obb_classes=None, obb_mask=None,
                labels=None, need_attention=False, **kwargs):
        """Run a forward pass through the model.

        Args:
            index:                tensor [B] int64 — sample indices (passed through, not used in computation)
            visual_features:      tensor [B, 30, 4096] float32 — EgoVLPv2 per-frame embeddings
            visual_mask:          tensor [B, 30] int32 — 1=valid frame, 0=padding
            env_grid:             tensor [B, 4096] int64 (static_env=True, env_class_averaging=False),
                                  [B, 30, 4096] int64 (temporal), or [B, 4096, K] int64
                                  (env_class_averaging=True, K=env_max_classes_per_voxel) — class
                                  index per voxel (0=empty/pad); 1–1220=boxer sem_id+1
            env_grid_mask:        tensor [B, 4096, K] bool or None — only used when
                                  env_class_averaging=True; 1=real class in that slot, 0=padding
            person_locations:     tensor [B, 30, 3] int32 or None
                                  voxel grid (ix, iy, iz) per second; 0=unknown/pad, 1–16=grid cell
            person_location_mask: tensor [B, 30] int32 or None — 1=valid, 0=padding
            trajectory_tokens:    tensor [B, N, trajectory_raw_dim] float32 or None
                                  raw per-track feature vectors (see trajectory_tokens.py); N is
                                  the PROVISIONAL fixed cap max_trajectory_tokens set in loader.py
            trajectory_mask:      tensor [B, N] int64 or None — 1=real token, 0=padding
            trajectory_classes:   tensor [B, N] int64 or None — class_embedding row per token
                                  (see object_class_vocab.py); 0 at padding slots
            trajectory_start_voxel: tensor [B, N, 3] int64 or None — only used when
                                  config.trajectory_simple=True; 0=pad, 1–16=grid cell per axis
            trajectory_end_voxel:  tensor [B, N, 3] int64 or None — same convention, track's
                                  end-of-window position
            trajectory_distance:  tensor [B, N] int64 or None — log-spaced bin of the distance
                                  travelled inside the window (trajectory_tokens.
                                  DISTANCE_BIN_EDGES); 0=pad, 1–16. Required under
                                  config.trajectory_token='end_distance', ignored otherwise
            gaze_voxels:          tensor [B, 30, 3] int64 or None — per second of the
                                  observation window, the voxel the actor was looking at
                                  (see gaze.py); same end-aligned padding as person_locations.
                                  0=pad/unknown, 1–16=grid cell per axis
            gaze_mask:            tensor [B, 30] int64 or None — 1=real, 0=padding
            obb_voxels:           tensor [B, N, 3] int64 or None — per fused 3D box, the grid
                                  index of its centre (see boxer_env.build_obb_tokens);
                                  0=pad, 1–16 per axis. N = config.max_obb_tokens
            obb_volumes:          tensor [B, N] int64 or None — log-spaced volume bin per box;
                                  0=pad, 1–16 (boxer_env.VOLUME_BIN_EDGES)
            obb_classes:          tensor [B, N] int64 or None — class_embedding row per box
                                  (Boxer sem_id+1, same vocabulary as env_grid); 0 at pad slots
            obb_mask:             tensor [B, N] bool/int64 or None — 1=real box, 0=padding
            labels:               dict or None — if provided, must contain:
                                    'interaction': tensor [B, 4096] int64 — binary voxel targets
            need_attention:       bool — return the voxel-query decoder's attention weights
                                  (config.decoder='voxel_query' only). DIAGNOSTIC ONLY: it
                                  materialises a [B, heads, 4096, L] tensor and disables the
                                  scaled_dot_product_attention fast path. See decoder.py.

        Returns:
            outputs: dict with:
                'interaction': tensor [B, 4096] float32 — raw (pre-sigmoid) logits
                'loss':        scalar tensor float32 — BCEWithLogitsLoss (only present when labels given)
                'attention':   tensor [B, 4096, L] float32 or None — only when need_attention
        """
        batch_size = visual_features.size(0)   # int — number of samples in the batch (e.g. 4)
        obs_time   = visual_features.size(1)   # int — sequence length in seconds (30)
        device     = visual_features.device

        # ---------------------------------------------------------------------------------
        # Step 1: Project visual features to transformer hidden dim
        # ---------------------------------------------------------------------------------
        # visual_mapped: tensor [B, 30, 768] float32
        # The LayerNorm is a no-op under the default config.token_layernorm=False.
        # It matters most here: nn.Linear's fan_in=4096 init leaves this segment at
        # norm 5.7 against obb's 66.9, purely because one was built by nn.Linear and
        # the other by nn.Embedding.
        # The marker is added AFTER _ln, so it enters at its own absolute magnitude
        # against a sqrt(H)-normalised token rather than being renormalised with the
        # content -- the same discipline the time code below follows. Under
        # type_embeddings='separate' video has no marker and this is the identity.
        visual_mapped = self._seg('video', self.visual_mapper(visual_features))

        # Time-to-anchor encoding, shared by every per-second modality so that
        # second n of video, of location and of gaze all carry the SAME delta --
        # which is what lets self-attention line them up in time. None when
        # config.time_encoding='none' (the default), in which case nothing below
        # changes and the forward pass stays bit-identical to the pre-change model.
        # [T, 768], broadcast over the batch.
        time_enc = self._time_encoding(obs_time, device, visual_mapped.dtype)
        if time_enc is not None:
            # delta is derived from the SLOT INDEX, which is only meaningful because
            # the loader end-aligns these sequences to obs_end. A caller passing a
            # different-length window would silently relabel every second.
            assert obs_time == self.config.observation_time, (
                f"time_encoding expects observation_time={self.config.observation_time} "
                f"tokens per second-modality, got {obs_time}; delta is read off the "
                "slot index, so a mismatch would mislabel every timestamp")
            visual_mapped = visual_mapped + time_enc

        # ---------------------------------------------------------------------------------
        # Step 2: Embed actor location along each spatial axis
        # ---------------------------------------------------------------------------------
        # Each axis index (0–16) is mapped to a 768-dim vector independently.
        # Using three separate embedding tables (rather than one 3-D combined one) keeps the
        # axis information factored and matches the FIction_orig implementation.
        zeros_loc = torch.zeros(batch_size, obs_time, self.config.transformer_hidden_size, device=device)
        # zeros_loc: tensor [B, 30, 768] — fallback if person_locations not provided
        #
        # config.location_tokens picks the layout:
        #   'axis'  (legacy) -- three tokens per second, one per axis, so a location
        #           token carries a SCALAR and the axes only combine downstream via
        #           attention. 90 tokens.
        #   'joint' -- one token per second carrying the whole position, through the
        #           same _voxel_pos_embedding every other positional token uses (so it
        #           inherits --position_encoding: the index-table sum under 'none', the
        #           Fourier code under 'fourier'). 30 tokens.
        # Neither dominates a priori. 'axis' gives attention three independent slots,
        # so a query for a high voxel can weight loc_z and ignore floor position;
        # 'joint' cannot separate the axes when routing, but binds them into one thing
        # instead of relying on the shared time stamp to say "these three are one
        # instant" -- a stamp that is only 2.6% of the token under the default
        # time_encoding_gain='fixed'.
        loc_joint = self.location_tokens == 'joint'
        if person_locations is not None:
            if loc_joint:
                loc_emb = self._seg(
                    'location', self._voxel_pos_embedding(person_locations))
                if time_enc is not None:
                    loc_emb = loc_emb + time_enc
                loc_x_emb = loc_y_emb = loc_z_emb = None
            else:
                # person_locations[:, :, 0]: tensor [B, 30] int — x-axis grid cell per second
                # One shared ln('location') across the three axis streams: they are one
                # segment fed in three parts, and giving each axis its own gamma/beta
                # would let the model rescale x against z for no stated reason.
                # Under 'unified' the same location marker goes on all three axis
                # streams -- they are one segment fed in three parts, exactly as the
                # gaze branch below does with its own marker. Legacy 'separate' has no
                # location marker in this layout (the per-axis tables are private to
                # location and gaze, so nothing collides), and _add_type is inert there.
                loc_x_emb = self._seg('location', self.location_embedding_x(person_locations[:, :, 0]))
                loc_y_emb = self._seg('location', self.location_embedding_y(person_locations[:, :, 1]))
                loc_z_emb = self._seg('location', self.location_embedding_z(person_locations[:, :, 2]))
                if time_enc is not None:
                    # The same delta on all three axis sequences: the x, y and z tokens
                    # of a given second describe one instant and must share its stamp.
                    loc_x_emb = loc_x_emb + time_enc
                    loc_y_emb = loc_y_emb + time_enc
                    loc_z_emb = loc_z_emb + time_enc
                loc_emb = None
        else:
            loc_x_emb = loc_y_emb = loc_z_emb = zeros_loc   # [B, 30, 768] each
            loc_emb = zeros_loc

        # ---------------------------------------------------------------------------------
        # Step 3: Embed the environment grid
        # ---------------------------------------------------------------------------------
        # Skipped entirely when env_grid is absent -- the 'obb' modality replaces
        # this segment, and a run using it need not carry a 4096-voxel grid at all.
        if env_grid is None:
            env_per_voxel = None
        elif self.config.env_class_averaging:
            # env_grid: [B, 4096, K] int64, env_grid_mask: [B, 4096, K] bool. Average the
            # class embeddings of every object overlapping a voxel (plain mean, not
            # confidence-weighted) instead of keeping only the highest-confidence winner.
            # Voxels with zero objects keep class_embedding row 0 (the learned "empty"
            # embedding) rather than being averaged with anything.
            embedded = self._class_embed(env_grid)                            # [B, 4096, K, 768]
            mask_f = env_grid_mask.float().unsqueeze(-1)                      # [B, 4096, K, 1]
            count = env_grid_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).float()  # [B, 4096, 1]
            avg_emb = (embedded * mask_f).sum(dim=2) / count                  # [B, 4096, 768]
            occupied = env_grid_mask.any(dim=2, keepdim=True)                 # [B, 4096, 1]
            # Row 0, "empty". Through _class_embed rather than .weight[0] so it stays
            # the same vector the lookups above produce under class_embedding_init.
            empty_emb = self._class_embed(
                torch.zeros((), dtype=torch.long, device=env_grid.device))    # [768]
            env_per_voxel = torch.where(occupied, avg_emb, empty_emb)         # [B, 4096, 768]
        elif self.config.use_static_env:
            # Static path: env_grid is [B, 4096] int64 — a single snapshot per video
            # class_embedding maps each integer class index to a 768-dim vector
            env_per_voxel = self._class_embed(env_grid)      # [B, 4096, 768]
        else:
            # Temporal path: env_grid is [B, 30, 4096] int64 — one snapshot per second
            # Embed every (second, voxel) pair, then average over the time dimension,
            # weighting only the non-empty time steps (class_idx > 0) to avoid diluting with zeros
            class_embedded = self._class_embed(env_grid)                 # [B, 30, 4096, 768]
            non_empty = (env_grid > 0).float()                           # [B, 30, 4096] float — 0 for empty voxels
            count_t = non_empty.sum(dim=1).clamp(min=1).unsqueeze(-1)    # [B, 4096, 1] — number of non-empty time steps per voxel
            # Weighted sum over time (only include non-empty voxels), divided by count
            env_per_voxel = (class_embedded * non_empty.unsqueeze(-1)).sum(dim=1) / count_t
            # env_per_voxel: [B, 4096, 768]

        if env_per_voxel is not None and self.config.env_positional_encoding:
            # Add each voxel's (ix,iy,iz) embedding (shared with the location/trajectory
            # modalities' per-axis tables) so the model can tell neighbouring voxels apart --
            # broadcasts [4096, 768] over the batch dimension.
            env_per_voxel = env_per_voxel + self._voxel_pos_embedding(self.env_voxel_ijk)

        if env_per_voxel is not None:
            env_per_voxel = self._ln('env', env_per_voxel)

        # ---------------------------------------------------------------------------------
        # Step 3b: Map trajectory tokens (only when supplied — see 'trajectory' modality below)
        # ---------------------------------------------------------------------------------
        # The continuous geometry/timing features go through the learned trajectory_mapper
        # MLP; the object's class goes through the SAME class_embedding table env tokens use
        # (added, not concatenated, so both stay in the shared 768-dim hidden space) — this
        # lets "what a sponge is" be one embedding learned from both modalities at once,
        # rather than training a second, separate class vocabulary from scratch. See
        # object_class_vocab.py for how a free-form track name resolves to a class_embedding
        # row (falling back to a dedicated OOV row for names the closed vocabulary can't match).
        if trajectory_tokens is not None and self.trajectory_token == 'end_distance':
            # --- 'end_distance' layout (see the __init__ block for the rationale) ---
            #   class + pos(end voxel) + distance bin + trajectory_type
            # No MLP and no raw feature vector: trajectory_tokens is read only for its
            # shape and dtype, so the segment is four table lookups wide. Padding slots
            # carry index 0 everywhere, which _voxel_pos_embedding maps to its pad
            # vector and the two nn.Embeddings to their row 0 -- and the pad mask below
            # keeps them out of attention regardless.
            if trajectory_end_voxel is None or trajectory_distance is None:
                raise ValueError(
                    "trajectory_token='end_distance' needs trajectory_end_voxel and "
                    "trajectory_distance; the loader emits both only under "
                    "trajectory_simple=True")
            trajectory_mapped = (self._voxel_pos_embedding(trajectory_end_voxel)
                                  + self.distance_embedding(trajectory_distance))
            if self.type_embeddings == 'separate':
                # Left at its original position in the sum, ahead of the class term,
                # so the legacy arm keeps its exact float32 association order.
                trajectory_mapped = trajectory_mapped + self.trajectory_type_embedding
            if trajectory_classes is not None:
                trajectory_mapped = trajectory_mapped + self._class_embed(trajectory_classes)
            trajectory_mapped = self._ln('trajectory', trajectory_mapped)
            if self.type_embeddings == 'unified':
                trajectory_mapped = trajectory_mapped + self._type('trajectory')
            trajectory_pad_mask = (~trajectory_mask.bool() if trajectory_mask is not None
                                    else torch.zeros(batch_size, trajectory_tokens.size(1),
                                                      dtype=torch.bool, device=device))
        elif trajectory_tokens is not None:
            trajectory_mapped = self.trajectory_mapper(trajectory_tokens)   # [B, N, 768]
            if trajectory_classes is not None:
                trajectory_mapped = trajectory_mapped + self._class_embed(trajectory_classes)
            if self.config.trajectory_simple and trajectory_start_voxel is not None:
                # Simplified path: no raw xyz in trajectory_tokens -- start/end positions are
                # embedded here via the SAME per-axis tables the location modality uses, so a
                # given voxel index means the same learned thing across both modalities.
                trajectory_mapped = (trajectory_mapped
                                      + self._voxel_pos_embedding(trajectory_start_voxel)
                                      + self._voxel_pos_embedding(trajectory_end_voxel))
            # Under 'separate' this layout has no marker (the MLP output is a private
            # term no other modality can produce), so _add_type is inert; 'unified'
            # gives it one for consistency with every other segment.
            trajectory_mapped = self._seg('trajectory', trajectory_mapped)
            trajectory_pad_mask = (~trajectory_mask.bool() if trajectory_mask is not None
                                    else torch.zeros(batch_size, trajectory_tokens.size(1),
                                                      dtype=torch.bool, device=device))
        else:
            trajectory_mapped = trajectory_pad_mask = None

        # ---------------------------------------------------------------------------------
        # Step 3c: Embed gaze tokens (only when supplied — see the 'gaze' modality below)
        # ---------------------------------------------------------------------------------
        # Purely positional: the loader already reduced each second's gaze to "this
        # was the voxel being looked at", so all that is left to encode is which
        # voxel, plus a marker saying the coordinate came from gaze. Padding seconds
        # carry index 0, which embeds to each table's row 0 exactly as an unknown
        # person location does, and are masked out of attention anyway.
        # config.gaze_tokens picks the layout, exactly as config.location_tokens does
        # for person_locations. gaze_mapped is a TUPLE either way -- three tensors
        # under 'axis', one under 'joint' -- so the assembly below stays layout-blind
        # and its mask/layout arithmetic follows len(gaze_mapped) rather than a
        # hardcoded 3.
        gaze_joint = self.gaze_tokens == 'joint'
        if gaze_voxels is not None:
            if gaze_joint:
                # One token per second carrying the whole looked-at voxel, through the
                # same _voxel_pos_embedding every other positional token uses -- so it
                # inherits --position_encoding (the index-table sum under 'none', the
                # shared Fourier code under 'fourier'). 90 tokens -> 30.
                #
                # This is what puts gaze inside --position_encoding's scope for the
                # first time. Under 'axis' a gaze token holds a SCALAR, so a code over
                # (i,j,k) has nowhere to go and gaze necessarily kept its own index-table
                # vocabulary while obb, location-joint and the decoder queries moved to
                # the Fourier one -- a third vocabulary nobody wanted.
                #
                # gaze_type_embedding is still required, and for a sharper reason than
                # under 'axis': a joint gaze token is otherwise byte-identical to a joint
                # LOCATION token at the same voxel, and to a decoder query there.
                # Padding is already handled -- gaze_voxels uses 0 for pad and
                # _voxel_pos_embedding maps any zero axis to spatial_pad_embedding.
                gaze_emb = self._seg('gaze', self._voxel_pos_embedding(gaze_voxels))
                if time_enc is not None:
                    gaze_emb = gaze_emb + time_enc
                gaze_mapped = (gaze_emb,)
            else:
                # One token per axis per second (3 x 30 = 90), exactly mirroring how
                # person_locations is fed in — same three embedding tables, same
                # per-axis split, same end-aligned padding. The gaze marker is added
                # because the encoder has no positional encoding over the sequence:
                # without it, a gaze-x token and a location-x token holding the same
                # index would be the identical vector, and self-attention could not
                # tell "the actor was here" from "the actor was looking here".
                gaze_x_emb = self._seg('gaze', self.location_embedding_x(gaze_voxels[:, :, 0]))
                gaze_y_emb = self._seg('gaze', self.location_embedding_y(gaze_voxels[:, :, 1]))
                gaze_z_emb = self._seg('gaze', self.location_embedding_z(gaze_voxels[:, :, 2]))
                if time_enc is not None:
                    # Gaze is the modality with most to gain here: without an order, a
                    # gaze sequence is close to a bag of voxels.
                    gaze_x_emb = gaze_x_emb + time_enc
                    gaze_y_emb = gaze_y_emb + time_enc
                    gaze_z_emb = gaze_z_emb + time_enc
                gaze_mapped = (gaze_x_emb, gaze_y_emb, gaze_z_emb)
            gaze_pad_mask = (~gaze_mask.bool() if gaze_mask is not None
                             else torch.zeros(batch_size, gaze_voxels.size(1),
                                              dtype=torch.bool, device=device))
        else:
            gaze_mapped = gaze_pad_mask = None

        # ---------------------------------------------------------------------------------
        # Step 3d: Embed OBB tokens (only when supplied — see the 'obb' modality below)
        # ---------------------------------------------------------------------------------
        # One token per fused 3D box: where it is + how big it is + what it is,
        # each through the table that already owns that information (see the
        # constructor). This replaces the 4096-token env grid with ~162 tokens
        # on average, and unlike that grid it can express a box's SIZE directly
        # rather than only through how many voxels the box happened to fill.
        if obb_voxels is not None:
            obb_mapped = self._seg('obb', self._voxel_pos_embedding(obb_voxels)
                                   + self._class_embed(obb_classes)
                                   + self.volume_embedding(obb_volumes))   # [B, N, 768]
            if obb_mask is not None:
                obb_pad_mask = ~obb_mask.bool()
                # --- OBB token dropout (config.obb_token_dropout), train only ---
                # The targeted regulariser for this model's failure mode. Generic
                # weight decay does not know what a kitchen is; this does. If the
                # model overfits by recognising the training kitchen from its object
                # inventory -- which is what a held-out PR-AUC peaking at step ~500
                # and decaying for the remaining 97% of the run looks like -- then
                # hiding a random share of the ~250 boxes every step makes any single
                # identifying configuration unreliable, while leaving the signal that
                # generalises ("there is a hob here, at this height") intact.
                # Applied BEFORE the all-pad guard below so that guard still runs.
                if self.training and self.obb_token_dropout > 0.0:
                    keep = torch.rand(obb_pad_mask.shape, device=obb_pad_mask.device)
                    obb_pad_mask = obb_pad_mask | (keep < self.obb_token_dropout)
                # 7–9% of real windows have NO Boxer boxes (missing CSV, or a
                # window past the end of one — measured in loader.py's startup
                # report). Masking every token of a segment makes that row's
                # attention softmax return NaN, which would poison the whole
                # batch. Keep slot 0 attendable on those rows: it is an
                # all-zero-index token, i.e. "pad voxel, pad size, empty class",
                # which is exactly the honest encoding of "no objects here".
                all_pad = obb_pad_mask.all(dim=1)              # [B]
                obb_pad_mask[all_pad, 0] = False
            else:
                obb_pad_mask = torch.zeros(batch_size, obb_voxels.size(1),
                                           dtype=torch.bool, device=device)
        else:
            obb_mapped = obb_pad_mask = None

        # ---------------------------------------------------------------------------------
        # Step 4: Assemble the full input sequence
        # ---------------------------------------------------------------------------------
        # Expand the learnable separator token to match batch size
        sep = self._ln('sep', self.sep_token).unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1)                                                   # [B, 1, 768]

        # Concatenate all segments along the sequence (dim=1) dimension. Modality controls which
        # segments are included; the unused ones are computed (small cost) but not concatenated.
        # modality is a comma-separated string of active modalities from
        # {'video', 'location', 'env', 'obb', 'gaze', 'trajectory'}.
        # Segments are always assembled in fixed order: video → location → env → obb →
        # gaze → trajectory, with a sep between each pair. (The encoder has no positional
        # encoding over the sequence, so this order is for readability; what does matter
        # is that 'video' stays first, since token 0 is what the output layer decodes.)
        #   'video'              → visual (30)                                              =   30
        #   'location'           → loc_x (30) | loc_y (30) | loc_z (30)                    =   90
        #                          or, under location_tokens='joint', one token/second   =   30
        #   'env'                → env (4096)                                               = 4096
        #   'obb'                → obb (250 cap, ~162 real)  -- the env alternative       =  250
        #   'gaze'               → gaze_x (30) | gaze_y (30) | gaze_z (30)              =   90
        #   'trajectory'         → trajectory (N, PROVISIONAL max_trajectory_tokens cap)
        #   'video,env'          → visual (30) | sep | env (4096)                           = 4127
        #   'video,location,env' → visual (30) | sep | loc_x..z (90) | sep | env (4096)    = 4219
        # ---------------------------------------------------------------------------------
        # Step 5: Build the padding mask for the transformer
        # ---------------------------------------------------------------------------------
        # In nn.TransformerEncoder, src_key_padding_mask has True where tokens should be IGNORED.
        # Shape must be [B, seq_len]; the transformer operates on the transposed [seq_len, B, H].
        sep_mask        = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)  # [B, 1] — sep tokens never masked
        visual_pad_mask = ~visual_mask.bool()   # [B, 30] bool — True where visual frame is padding
        loc_pad_mask    = (~person_location_mask.bool()
                           if person_location_mask is not None
                           else torch.zeros(batch_size, obs_time, dtype=torch.bool, device=device))
        # [B, 30] bool — True where actor location is unknown (pad token)
        env_pad_mask    = (torch.zeros(batch_size, env_per_voxel.size(1),
                                       dtype=torch.bool, device=device)
                           if env_per_voxel is not None else None)
        # [B, 4096] bool — all False: every voxel slot is a real input (empty voxels have zero embedding)

        modality_raw = getattr(self.config, 'input_modality', 'video,location,env')
        active = set(modality_raw.split(','))

        segments_seq  = []
        segments_mask = []
        # (name, n_tokens) as actually assembled, so analysis/voxel_query_attention.py
        # can attribute a decoder attention column to a modality without
        # re-deriving this layout and drifting from it. Returned only under
        # need_attention; building the list costs nothing.
        segment_layout = []
        for mod in ('video', 'location', 'env', 'obb', 'gaze', 'trajectory'):
            if mod not in active:
                continue
            if segments_seq:  # sep only between two present segments, never leading
                segments_seq.append(sep)
                segments_mask.append(sep_mask)
                segment_layout.append(('sep', 1))
            if mod == 'video':
                segments_seq.append(visual_mapped)
                segments_mask.append(visual_pad_mask)
                segment_layout.append(('video', visual_mapped.size(1)))
            elif mod == 'location':
                if loc_joint:
                    segments_seq.append(loc_emb)
                    segments_mask.append(loc_pad_mask)
                    segment_layout.append(('location', loc_emb.size(1)))
                else:
                    segments_seq.extend([loc_x_emb, loc_y_emb, loc_z_emb])
                    segments_mask.extend([loc_pad_mask, loc_pad_mask, loc_pad_mask])
                    segment_layout.append(('location', 3 * loc_x_emb.size(1)))
            elif mod == 'env':
                if env_per_voxel is None:
                    raise ValueError(
                        "input_modality includes 'env' but env_grid was not provided")
                segments_seq.append(env_per_voxel)
                segments_mask.append(env_pad_mask)
                segment_layout.append(('env', env_per_voxel.size(1)))
            elif mod == 'obb':
                if obb_mapped is None:
                    raise ValueError(
                        "input_modality includes 'obb' but obb_voxels was not provided")
                segments_seq.append(obb_mapped)
                segments_mask.append(obb_pad_mask)
                segment_layout.append(('obb', obb_mapped.size(1)))
            elif mod == 'gaze':
                if gaze_mapped is None:
                    raise ValueError(
                        "input_modality includes 'gaze' but gaze_voxels was not provided")
                # len(gaze_mapped) is 3 under gaze_tokens='axis' and 1 under 'joint';
                # every token in the tuple is the same 30 seconds, so one mask serves all.
                segments_seq.extend(gaze_mapped)
                segments_mask.extend([gaze_pad_mask] * len(gaze_mapped))
                segment_layout.append(('gaze', len(gaze_mapped) * gaze_mapped[0].size(1)))
            elif mod == 'trajectory':
                if trajectory_mapped is None:
                    raise ValueError(
                        "input_modality includes 'trajectory' but trajectory_tokens was not provided")
                segments_seq.append(trajectory_mapped)
                segments_mask.append(trajectory_pad_mask)
                segment_layout.append(('trajectory', trajectory_mapped.size(1)))

        input_sequence = torch.cat(segments_seq, dim=1)
        attention_mask = torch.cat(segments_mask, dim=1)

        # ---------------------------------------------------------------------------------
        # Step 6: Run the transformer encoder
        # ---------------------------------------------------------------------------------
        # nn.TransformerEncoder with batch_first=False requires [seq_len, B, H]
        input_sequence = input_sequence.permute(1, 0, 2)   # [4219, B, 768]

        # --readout cls: prepend the dedicated summary token. Done here, after the
        # permute, so the concatenation is in the encoder's own [L, B, H] layout;
        # the key-padding mask stays batch-first ([B, L]) and gains a False column,
        # since the CLS seat is never padding. Every token index shifts by one --
        # transformer_output[0] is the only position-dependent read in the repo,
        # and it wants the CLS seat under this setting anyway.
        if self.readout == 'cls':
            # std=0.02 init puts this at norm 0.56 -- the quietest token in the
            # sequence, and the one the decoder is meant to read as the global
            # summary. ln('cls') is the flag's single largest relative change.
            cls = self._ln('cls', self.cls_token).expand(-1, batch_size, -1)   # [1, B, 768]
            input_sequence = torch.cat([cls, input_sequence], dim=0)  # [1+L, B, 768]
            segment_layout.insert(0, ('cls', 1))
            attention_mask = torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
                attention_mask,
            ], dim=1)                                               # [B, 1+L]

        transformer_output = self.transformer(input_sequence, src_key_padding_mask=attention_mask)
        # transformer_output: [4219, B, 768] — contextualised representation of every token

        # ---------------------------------------------------------------------------------
        # Step 7: Aggregate and decode
        # ---------------------------------------------------------------------------------
        attn_weights = None
        if self.decoder is not None:
            # Coordinate-query decoder: no aggregation step at all. Every one of the
            # 4096 voxels queries the whole sequence for itself, and the SAME padding
            # mask the encoder used is handed over unchanged (rebuilding it here
            # would be a silent correctness bug).
            logits, attn_weights = self.decoder(
                transformer_output, attention_mask, need_weights=need_attention)
        else:
            # Seat 0 under both readouts: the first visual frame ('token0', legacy) or
            # the learned CLS token prepended just above ('cls').
            aggregate_feature = transformer_output[0]   # [B, 768]

            # Project to the output space: one logit per voxel
            logits = self.output_layer(aggregate_feature)   # [B, 4096] float32 — raw scores, not probabilities

        outputs = {"interaction": logits}   # dict with tensor [B, 4096]
        if need_attention:
            # [B, 4096, L], averaged over heads -- for analysis/voxel_query_attention.py
            # only. Never request this during training: see decoder.py's docstring.
            outputs["attention"] = attn_weights
            # [(name, start, end)] over the same L, including 'sep' and 'cls'.
            spans, at = [], 0
            for name, n in segment_layout:
                spans.append((name, at, at + n))
                at += n
            outputs["segments"] = spans
        if labels is not None:
            # BCEWithLogitsLoss applies sigmoid internally before computing cross-entropy,
            # so logits are passed directly (no manual sigmoid needed)
            # labels['interaction']: tensor [B, 4096] int64 — cast to float for BCE
            target = labels['interaction'].float()
            if getattr(self, 'loss', 'bce') == 'focal':
                loss = sigmoid_focal_loss(logits, target,
                                          self.focal_gamma, self.focal_alpha)
            else:
                loss = self.loss_func(logits, target)
            outputs["loss"] = loss   # scalar tensor

        return outputs
