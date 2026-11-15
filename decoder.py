"""Coordinate-query cross-attention decoder (docs/voxel_query_decoder_spec.md §4).

The readout this replaces was a single ``nn.Linear(768, 4096)`` applied to one
token of the encoder output. That layer holds an independent 768-vector per
voxel: 3.15M parameters in which voxel (5,8,6) shares nothing with (5,8,7), and
no mechanism for routing evidence about a *region* to the voxels in it. Under
participant-level splits (one kitchen per participant) that is a large capacity
sink sitting exactly where memorising training-kitchen voxel priors would pay.

This inverts the relationship. The voxel coordinate becomes the **query**, the
encoder's output tokens are keys/values, and a single ``nn.Linear(768, 1)``
**shared by all 4096 voxels** maps each attended vector to its logit. A voxel's
prediction must then come from evidence it retrieved for that location, and the
attention map says which tokens those were -- a far sharper probe of "is this
anticipating, or reading a co-occurrence prior?" than an ablation delta.

Three things here are deliberately not free parameters of the architecture:

  * ``voxel_coords`` is C order over (16,16,16), ``flat = i*256 + j*16 + k``.
    That is the layout ``labels['interaction']`` uses (loader.py builds it as
    ``gt[coord_to_voxel(...)] = 1`` over a grid flattened with ``reshape(-1,3)``
    from ``meshgrid(indexing='ij')``). A mismatch here trains happily against a
    permuted target and is invisible in the loss -- smoke_decoder.py tests 9 and
    11 exist for exactly this.
  * ``voxel_bias`` defaults to **factorised** (48 parameters, one vector per
    axis) rather than a flat [4096]. The production grids are wall-fitted per
    participant, but nothing fixes which wall is +x -- eight rectangle
    symmetries remain, and the long axis really is inconsistent (4 participants
    x-long, 5 y-long). So absolute horizontal position does not correspond
    across kitchens while height does, and a factorised bias can express the
    part that transfers ("interactions happen at counter level") without the
    capacity to memorise one kitchen's floor plan. ``full`` is kept as an
    ablation arm; if it wins, that reasoning is wrong and worth chasing.
    ``mlp`` sits between the two: a small Tanh network of (x,y,z) can express
    genuine cross-axis structure ("this corner", not just "this wall") that
    factorised's pure sum cannot, while its size and smooth activation keep it
    far short of full's per-voxel memorisation capacity. Only meaningful
    combined with a scalar init (zero/prior) and ``voxel_bias_lr > 0`` -- it
    has no closed-form fit for an arbitrary field, so it has to learn its
    shape rather than being handed one.
  * ``decoder_residual`` defaults to **raw** (``h = q + attn_out``), the standard
    pre-LN residual and what every run before 2026-08-11 used. Standard, but the
    usual justification does not hold here: in a normal transformer both sides of
    a residual carry comparable magnitudes, whereas ``q`` comes from nn.Embedding
    tables (N(0,1) init, three rows summed) while ``attn_out`` is an attention
    output. Measured on a trained axis_sum checkpoint: |q| = 55.7 against
    |attn_out| = 1.11, a 50:1 ratio -- so the vector reaching ln_o points almost
    exactly where the query points, and evidence arrives as a small perturbation
    the head has to amplify. It does reach the output (26-39% of logit variance
    across the LOOCV folds is episode-driven), so this is a conditioning concern,
    not a blockage. ``normed`` (``ln_q(q) + attn_out``) halves the ratio to 25:1;
    ``balanced`` (``ln_q(q) + ln_a(attn_out)``) puts both sides at sqrt(D) and is
    the arm that actually tests "what if evidence entered at the query's scale".

    Those three all keep ``q`` in the output and only change how loud it is, and on
    fold P01 they land on top of each other (raw .0463, normed .0460, balanced
    .0495) -- turning the query down is measurably not the variable. Two further
    arms remove it instead, for a different question. Measured on the P08 baseline
    at its best checkpoint, **91% of logit variance is static per voxel** (bit for
    bit identical on every episode) and only 9% is episode-driven; of that static
    part the frozen voxel bias is 0.23%, so essentially all of it arrives through
    ``q``. What nobody knows is whether that map is stored in THIS RESIDUAL or in
    the retrieved evidence itself -- the kitchen does not move, so a voxel attends
    to the same boxes every episode and ``attn_out`` can be near-constant without
    anything being wired wrong. The two cases need opposite fixes.

      * ``evidence`` (``h = ln_a(attn_out)``) cuts the residual entirely. ``q``
        still selects what to attend to, but a voxel's logit becomes a function of
        what it retrieved rather than of which voxel it is. If the static map
        survives this, it was never in the residual.
      * ``concat`` (``h = [ln_q(q); ln_a(attn_out)]``, head 1536->1) keeps both but
        stops summing them, so the two halves of ``head.weight`` say how much the
        model wanted each -- unrecoverable from a sum, where the two are added into
        one vector before the head ever sees it. Incompatible with ``decoder_ffn``,
        which is built at d_model; that combination raises at construction.

  * ``need_weights=False`` on the training path. nn.MultiheadAttention only
    dispatches to scaled_dot_product_attention when weights are not requested;
    ask for them and it materialises [B, heads, 4096, L] -- ~1.1 GB at B=16 on
    the obb branch, ~8.7 GB on env. Only analysis/voxel_query_attention.py sets
    it True, and only on a small batch.
"""
import math

import numpy as np
import torch
from torch import nn


def fourier_scale_per_axis(value):
    """(sx, sy, sz) from a --voxel_query_fourier_scale value.

    Accepts a single number -- the legacy form, all three axes alike -- or three
    comma-separated numbers. Per-axis exists because the production grids are not
    cubic and not cubic in the same way: measured over the nine
    ``new_voxel_centers/*.npy``, the z cell is 13.3-14.6 cm in every kitchen while
    x runs 16.8-41.7 cm and y 12.6-46.6 cm. One shared init frequency therefore
    resolves height far more finely than either floor axis, and W_r can learn its
    way out of that but the init cannot.

    Lives here rather than in model.py because model.py imports this module.
    """
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    parts = [p.strip() for p in str(value).split(',') if p.strip()]
    if len(parts) == 1:
        return (float(parts[0]),) * 3
    if len(parts) != 3:
        raise ValueError(
            f"fourier scale must be one value or three comma-separated ones, got {value!r}")
    return tuple(float(p) for p in parts)


def mlp_hidden_sizes(value):
    """[h1, h2, ...] from a --voxel_bias_mlp_hidden value ("32,32" -> [32, 32]).

    Lives here, not model.py, for the same reason fourier_scale_per_axis does:
    model.py imports this module and calls this at CustomConfig construction
    time so a malformed value fails at __init__ rather than at the first
    forward pass.
    """
    parts = [p.strip() for p in str(value).split(',') if p.strip()]
    if not parts:
        raise ValueError(f"voxel_bias_mlp_hidden must list at least one hidden "
                         f"width, got {value!r}")
    sizes = [int(p) for p in parts]
    if any(h <= 0 for h in sizes):
        raise ValueError(f"voxel_bias_mlp_hidden widths must be positive, got {value!r}")
    return sizes


def fourier_code(ijk, W_r, gain, grid):
    """[..., 2*R] learnable-Fourier code of a voxel centre (LLaVA-4D Eq. 5).

    ONE implementation, called by both sides. model._spatial_encoding routes the
    input tokens through it and the decoder's untied queries call it directly, so
    "tied vs untied" differs in exactly one thing -- whose parameters -- rather
    than in two call sites agreeing on a formula. That agreement is precisely
    what smoke_position_encoding.py test 3 exists to police, and a second copy of
    these four lines is how it would silently stop holding.

    ijk: LongTensor [..., 3], 1..16 per axis (0 = pad, handled by the CALLER --
        queries are never padded, input tokens are).
    W_r: [R, 3] frequencies.  gain: scalar tensor, the code's norm.
    Returns a vector of norm exactly `gain`: cos^2+sin^2 = 1 per frequency makes
    the raw row norm sqrt(R) whatever W_r holds, so the division is exact.
    """
    coords = (ijk.to(W_r.dtype) - 0.5) / grid          # [..., 3] in (0,1)
    proj = coords @ W_r.T                              # [..., R]
    code = torch.cat([proj.cos(), proj.sin()], dim=-1)  # [..., 2R]
    return code / math.sqrt(W_r.shape[0]) * gain


class VoxelQueryDecoder(nn.Module):
    """4096 coordinate queries cross-attending to the encoder's output tokens.

    Args:
        config: CustomConfig -- reads output_dim, transformer_hidden_size,
            voxel_query_type, voxel_bias, voxel_bias_mlp_hidden, decoder_heads,
            decoder_ffn, decoder_ffn_dim, voxel_query_fourier_scale,
            location_vocab_size.
        location_embeddings: the model's (x, y, z) nn.Embedding tables, shared
            rather than duplicated for voxel_query_type='axis_sum'. Passing None
            is only valid for the other two variants.
        pos_embedding: the model's ``_voxel_pos_embedding`` bound method, for
            voxel_query_type='fourier' under config.position_encoding='fourier'.
            Passing it is what makes a query and an input token at one voxel carry
            the identical code; passing None keeps the private-W_r query of every
            fourier run before that flag existed.
    """

    def __init__(self, config, location_embeddings=None, pos_embedding=None):
        super().__init__()
        d_model = config.transformer_hidden_size
        self.num_voxels = config.output_dim
        grid = round(self.num_voxels ** (1 / 3))
        if grid ** 3 != self.num_voxels:
            raise ValueError(
                f"output_dim={self.num_voxels} is not a cube; the decoder queries "
                "one voxel per output and needs a (N,N,N) grid")
        self.grid = grid
        self.query_type = config.voxel_query_type
        # getattr: a config.json written before this landed has no such key
        # and must build the tied/legacy model rather than crash.
        self.untied = bool(getattr(config, 'voxel_query_untied', False))
        # How many times the ONE block below is applied. See forward().
        self.iters = int(getattr(config, 'decoder_iters', 1))
        self.bias_type = config.voxel_bias

        # --- Query coordinates, in the label grid's own flattening order ---
        # ijk[flat] = (i, j, k) for flat = i*N^2 + j*N + k. np.unravel_index over
        # (N,N,N) is C order, the same call model.py already uses to build
        # env_voxel_ijk, so env tokens and voxel queries agree by construction.
        ijk = np.stack(np.unravel_index(np.arange(self.num_voxels),
                                        (grid, grid, grid)), axis=-1)  # (4096,3), 0..15
        self.register_buffer('voxel_ijk0', torch.as_tensor(ijk, dtype=torch.long))
        # 1..16, matching the location tables' convention (0 = pad/unknown).
        self.register_buffer('voxel_ijk1', torch.as_tensor(ijk + 1, dtype=torch.long))
        # Cell centres in (0,1) per axis -- normalised, because the grid is
        # normalised per participant and metres are not comparable across kitchens.
        self.register_buffer('voxel_coords',
                             torch.as_tensor((ijk + 0.5) / grid, dtype=torch.float32))

        # --- Query construction ---
        if self.query_type == 'axis_sum':
            if location_embeddings is None:
                raise ValueError("voxel_query_type='axis_sum' needs the model's "
                                 "location embedding tables")
            # Held in a tuple, NOT assigned as submodules. nn.Module only registers
            # a Module assigned directly to an attribute, so this keeps the tables
            # out of the decoder's state_dict -- they belong to the parent model and
            # already appear there once. Registering them twice makes save_pretrained
            # refuse to write safetensors ("shared tensors ... mismatching the
            # transformers base configuration"), i.e. every axis_sum run would die at
            # its first checkpoint. Autograd is unaffected: same objects, one set of
            # parameters, gradients from queries and input tokens accumulate together.
            self._location_embeddings = tuple(location_embeddings)
            # Same role as gaze_type_embedding / obb_type_embedding: with no
            # positional encoding, a query built from the shared location tables
            # would otherwise be the identical vector to an *input* location
            # token at that voxel. Only axis_sum needs it -- the other two
            # variants have private parameterisations already.
            self.query_type_embedding = nn.Parameter(torch.randn(d_model))
        elif self.query_type == 'fourier':
            # Learnable Fourier features (LLaVA-4D Eq. 5): continuous in the
            # coordinate, so nearby voxels start out similar, and W_r mixes the
            # three axes before cos/sin so the code is not merely additive in them.
            # fourier_scale is the sensitive knob: too small and all 4096 queries
            # collapse toward one vector (the model degenerates to a single global
            # logit); too large and adjacent voxels alias.
            self._pos_embedding = None
            if pos_embedding is not None:
                # Shared path. Held in a tuple for the same reason
                # _location_embeddings is: it is a bound method of the parent model
                # and registering it here would duplicate the parent's parameters in
                # this module's state_dict. The queries are then literally the
                # function the OBB/env/trajectory tokens go through.
                self._pos_embedding = (pos_embedding,)
                # Now REQUIRED. The comment this replaces said the non-axis_sum
                # variants need no marker because their parameterisation is private
                # -- true only while it was private. Sharing the code is exactly what
                # makes a bare query identical to an input token's positional part,
                # which is the collision gaze_type_embedding and obb_type_embedding
                # exist to prevent.
                self.query_type_embedding = nn.Parameter(torch.randn(d_model))
            elif self.untied:
                # UNTIED: the same code as the shared path in every respect --
                # same formula (fourier_code), same per-axis init, same gain, same
                # marker -- except the parameters are the decoder's own. That is
                # what makes tied-vs-untied a one-variable comparison instead of
                # the 2026-08-11 arm, which changed the basis, the tying AND the
                # amplitude at once (its norm was 0.707 against axis_sum's 55.7,
                # a 0.6:1 query:evidence regime against a 50:1 one).
                #
                # This is also PETR's arrangement (ECCV 2022): coordinate-derived
                # queries and position-aware keys built by separate networks. The
                # tied path is DAB-DETR's argument instead -- put the query in the
                # key's positional space -- which DAB-DETR could take for free
                # because its sinusoids have no parameters to share.
                self.W_r = nn.Parameter(torch.empty(d_model // 2, 3))
                for axis, std in enumerate(fourier_scale_per_axis(
                        config.voxel_query_fourier_scale)):
                    nn.init.normal_(self.W_r[:, axis], std=std)
                # torch.tensor draws no random numbers, so adding this consumes no
                # RNG and every other parameter's init is unchanged by the flag.
                self.query_gain = nn.Parameter(
                    torch.tensor(float(config.position_encoding_gain_init)))
                self.query_type_embedding = nn.Parameter(torch.randn(d_model))
            else:
                # Legacy private W_r: one scale over three anisotropic axes, no
                # marker, norm pinned at 0.707. Kept bit-identical so the nine
                # 2026-08-11 vq_fourier checkpoints still load and still reproduce.
                self.W_r = nn.Parameter(torch.empty(d_model // 2, 3))
                sx, sy, sz = fourier_scale_per_axis(config.voxel_query_fourier_scale)
                if sx == sy == sz:
                    # ONE init call, as before: filling per column would consume the
                    # RNG differently and the 2026-08-11 checkpoints' weights with it.
                    nn.init.normal_(self.W_r, std=sx)
                else:
                    for axis, std in enumerate((sx, sy, sz)):
                        nn.init.normal_(self.W_r[:, axis], std=std)
        elif self.query_type == 'learned':
            # Control arm: 3.15M parameters, the same count as the layer being
            # removed, and no coordinate structure at all. If this matches
            # axis_sum then the gain came from the cross-attention routing rather
            # than from queries being coordinates -- worth knowing either way.
            self.query_embedding = nn.Embedding(self.num_voxels, d_model)
        else:
            raise ValueError(f"Unknown voxel_query_type: {self.query_type!r}")

        # --- The block itself: pre-LN, one cross-attention layer, optional FFN ---
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_m = nn.LayerNorm(d_model)
        # ln_o is built further down, once the residual mode is known: 'concat'
        # makes it twice as wide. Safe to move because nn.LayerNorm's init draws no
        # random numbers, so the RNG the attention and head consume is unchanged --
        # smoke_decoder.py test 1 (bit-identical regression) is what proves it.
        # dropout defaults to 0.0 -- nn.MultiheadAttention's own default, and what
        # every run before this flag used. Note the asymmetry it leaves: the encoder
        # has carried 0.1 all along while the 4096 queries attend with none.
        self.attn = nn.MultiheadAttention(d_model, config.decoder_heads,
                                          dropout=getattr(config, 'decoder_dropout', 0.0),
                                          batch_first=False)  # matches the encoder
        # See the module docstring. ln_a does not exist under raw/normed, so those
        # checkpoints keep exactly the state_dict they had before this flag existed
        # and cross-loading between arms fails loudly rather than silently.
        # nn.LayerNorm's init draws no random numbers, so building it consumes no
        # RNG and the other arms' weights are unaffected.
        self.residual = getattr(config, 'decoder_residual', 'raw')
        if self.residual not in ('raw', 'normed', 'balanced', 'evidence', 'concat'):
            raise ValueError(
                f"Unknown decoder_residual: {self.residual!r} (expected 'raw', "
                "'normed', 'balanced', 'evidence' or 'concat')")
        if self.residual in ('balanced', 'evidence', 'concat'):
            self.ln_a = nn.LayerNorm(d_model)
        # 'concat' keeps both sides side by side instead of summing them, so
        # everything downstream of h is twice as wide.
        head_dim = d_model * 2 if self.residual == 'concat' else d_model
        self.ln_o = nn.LayerNorm(head_dim)
        self.use_ffn = bool(config.decoder_ffn)
        if self.use_ffn:
            if self.residual == 'concat':
                # The FFN is built at d_model and h is 2*d_model here, so the two
                # cannot coexist. Caught at construction rather than deep in the
                # forward pass, where the shape error names neither flag.
                raise ValueError(
                    "decoder_ffn is not supported with decoder_residual='concat' "
                    f"(the FFN is {d_model}-wide, h is {head_dim}-wide). Turn one off.")
            # Off by default: 4.72M parameters makes the model LARGER than the
            # layer this replaces, which undercuts the whole argument. Report the
            # parameter count alongside the metric if you turn it on.
            self.ln_f = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, config.decoder_ffn_dim), nn.GELU(),
                nn.Linear(config.decoder_ffn_dim, d_model))
        # ONE head for all 4096 voxels. This is the parameter reduction that
        # matters; a per-voxel head would be the layer we are removing.
        self.head = nn.Linear(head_dim, 1)

        # --- Unconditional per-voxel prior ---
        # Kept in some form deliberately: the 3.15M layer could express a per-voxel
        # prior, and under ~1:200 imbalance that is doing real work, so removing it
        # outright would confound the comparison.
        if self.bias_type == 'factorised':
            self.bias_x = nn.Parameter(torch.zeros(grid))
            self.bias_y = nn.Parameter(torch.zeros(grid))
            self.bias_z = nn.Parameter(torch.zeros(grid))
        elif self.bias_type == 'full':
            self.voxel_bias = nn.Parameter(torch.zeros(self.num_voxels))
        elif self.bias_type == 'mlp':
            # A small function of (x,y,z) together: unlike 'factorised' it can
            # express genuine cross-axis structure ("this CORNER", not just
            # "this whole wall"), and unlike 'full' it is too small and -- via
            # Tanh's smooth, low-pass-biased curvature (Tancik et al., NeurIPS
            # 2020) -- too smooth to memorise one training kitchen's exact
            # per-voxel layout. Raw (x,y,z) in, deliberately NOT Fourier-coded:
            # that spectral bias toward smooth outputs is the point, not a
            # limitation to engineer around.
            sizes = mlp_hidden_sizes(getattr(config, 'voxel_bias_mlp_hidden', '32,32'))
            layers, d_in = [], 3
            for h in sizes:
                layers += [nn.Linear(d_in, h), nn.Tanh()]
                d_in = h
            layers.append(nn.Linear(d_in, 1))
            self.bias_mlp = nn.Sequential(*layers)
            # Zero the LAST layer only: output = 0*hidden + bias = a constant,
            # for every voxel, regardless of the random hidden-layer init. The
            # same "flat until set_prior_logit says otherwise" contract
            # factorised/full keep via torch.zeros(...) above -- an MLP can't be
            # zero-initialised whole (every hidden unit would be dead) but can
            # be made to START flat this way, with hidden weights already in
            # place and ready to move under --voxel_bias_lr > 0.
            nn.init.zeros_(self.bias_mlp[-1].weight)
            nn.init.zeros_(self.bias_mlp[-1].bias)
        else:
            raise ValueError(f"Unknown voxel_bias: {self.bias_type!r}")

    # -- prior init (Stage B) -------------------------------------------------

    def set_prior_logit(self, prior_logit):
        """Start the bias at the class prior. Split three ways under
        'factorised' so the SUM starts at prior_logit, not 3x it. Under 'mlp'
        this only touches the (already-zeroed) last layer's bias -- see
        __init__ -- so the hidden layers' random init is untouched."""
        with torch.no_grad():
            if self.bias_type == 'factorised':
                for p in (self.bias_x, self.bias_y, self.bias_z):
                    p.fill_(float(prior_logit) / 3.0)
            elif self.bias_type == 'mlp':
                self.bias_mlp[-1].bias.fill_(float(prior_logit))
            else:
                self.voxel_bias.fill_(float(prior_logit))

    def set_voxel_bias(self, values):
        """Start the bias at an arbitrary per-voxel field (Stage B, spatial arms).

        ``values`` is a scalar (delegates to set_prior_logit) or a [num_voxels]
        array in the label flattening order, flat = i*256 + j*16 + k -- the same
        order ``bias()`` reads and the same one ``labels['interaction']`` uses.
        Passing it in any other order trains happily against a permuted field and
        is invisible in the loss; smoke_decoder.py test 11 is what pins it.

        Under 'full' the field is copied verbatim. Under 'factorised' only 48
        numbers exist, so the field is projected onto the closest additive one --
        the least-squares main-effects fit

            a[i] = mean_jk v - (2/3) v_bar,  and likewise b[j], c[k]

        whose sum reproduces v exactly when v really is additive (constant
        included: 3g - 2g = g) and otherwise keeps its axis marginals while
        dropping every interaction between axes. That is not a lossy
        implementation detail to work around, it IS the arm: 'factorised' exists
        to carry the part of a spatial prior that transfers across kitchens
        (height) and structurally cannot carry a memorised floor plan. Use
        --voxel_bias full when the point is to hand the model the whole field.

        Returns (r2, captured, total) so the caller can log how much of the field
        survived the projection -- 1.0 under 'full' by construction.
        """
        v = torch.as_tensor(values, dtype=torch.float32).flatten()
        if v.numel() == 1:
            self.set_prior_logit(float(v.item()))
            return 1.0, 0.0, 0.0
        if self.bias_type == 'mlp':
            raise ValueError(
                "voxel_bias='mlp' has no closed-form fit for an arbitrary per-voxel "
                "field -- 'factorised' projects onto its additive part and 'full' "
                "copies it verbatim, but 'mlp' would need its own pretraining loop. "
                "Use --voxel_bias_init zero or prior (both scalar) and "
                "--voxel_bias_lr > 0 to let it learn instead.")
        if v.numel() != self.num_voxels:
            raise ValueError(
                f"voxel bias field has {v.numel()} entries, expected a scalar or "
                f"{self.num_voxels} (one per voxel, flat = i*256 + j*16 + k)")
        with torch.no_grad():
            if self.bias_type == 'full':
                self.voxel_bias.copy_(v)
                return 1.0, float(v.var(unbiased=False)), float(v.var(unbiased=False))
            g = v.view(self.grid, self.grid, self.grid)
            grand = g.mean()
            mx, my, mz = g.mean((1, 2)), g.mean((0, 2)), g.mean((0, 1))
            self.bias_x.copy_(mx - 2.0 / 3.0 * grand)
            self.bias_y.copy_(my - 2.0 / 3.0 * grand)
            self.bias_z.copy_(mz - 2.0 / 3.0 * grand)
            fit = self.bias().detach()
            total = float(v.var(unbiased=False))
            resid = float((v - fit).var(unbiased=False))
            r2 = 1.0 - resid / total if total > 0 else 1.0
            return r2, total - resid, total

    # -- queries and bias -----------------------------------------------------

    def queries(self):
        """[4096, D] -- one query vector per voxel, row `flat` for voxel
        (i, j, k) with flat = i*256 + j*16 + k."""
        if self.query_type == 'axis_sum':
            emb_x, emb_y, emb_z = self._location_embeddings
            return (emb_x(self.voxel_ijk1[:, 0])
                    + emb_y(self.voxel_ijk1[:, 1])
                    + emb_z(self.voxel_ijk1[:, 2])
                    + self.query_type_embedding)
        if self.query_type == 'fourier':
            if self._pos_embedding is not None:
                # The parent's own code, on the same 1..16 indices its input tokens
                # use. voxel_ijk1 = voxel_ijk0 + 1 and _spatial_encoding divides by
                # (ijk - 0.5), so this lands on voxel_coords by construction --
                # smoke_position_encoding.py test 3 pins that they agree.
                return self._pos_embedding[0](self.voxel_ijk1) + self.query_type_embedding
            if self.untied:
                # voxel_ijk1 is 1..16 and fourier_code subtracts 0.5, so this lands
                # on voxel_coords exactly as the tied path does -- same grid
                # convention, different parameters.
                return (fourier_code(self.voxel_ijk1, self.W_r,
                                     self.query_gain, self.grid)
                        + self.query_type_embedding)
            proj = self.voxel_coords @ self.W_r.T                    # [4096, D/2]
            return (torch.cat([proj.cos(), proj.sin()], dim=-1)
                    / math.sqrt(self.W_r.shape[0] * 2))
        return self.query_embedding.weight

    def bias(self):
        """[4096] additive per-voxel prior, in the label flattening order."""
        if self.bias_type == 'factorised':
            return (self.bias_x[self.voxel_ijk0[:, 0]]
                    + self.bias_y[self.voxel_ijk0[:, 1]]
                    + self.bias_z[self.voxel_ijk0[:, 2]])
        if self.bias_type == 'mlp':
            return self.bias_mlp(self.voxel_coords).squeeze(-1)
        return self.voxel_bias

    # -- forward --------------------------------------------------------------

    def forward(self, memory, memory_key_padding_mask=None, need_weights=False):
        """
        Args:
            memory: [L, B, D] encoder output (sequence-first, as the encoder
                produces it -- no transposing back and forth).
            memory_key_padding_mask: [B, L] bool, True = pad. Must be the very
                same mask the encoder was given; rebuilding it here would be a
                silent correctness bug. Every row needs at least one False or
                the softmax returns NaN -- true in practice because the video
                segment is always present, and asserted in the smoke tests.
            need_weights: leave False for training (see the module docstring).

        Returns:
            logits: [B, 4096]
            attn:   [B, 4096, L] averaged over heads, or None
        """
        batch_size = memory.shape[1]
        q = self.queries().unsqueeze(1).expand(-1, batch_size, -1)   # [4096, B, D]
        m = self.ln_m(memory)

        if self.iters > 1:
            # --- Residual stream, the way a residual is actually meant to work ---
            # The critique this answers: h = q + attn_out is a textbook residual by
            # construction, but it does none of a residual's usual work. A residual
            # exists so a representation built up over DEPTH is not destroyed and
            # gradients still reach the early layers. Here there was one layer and q
            # was a fixed function of (i,j,k) -- identical every episode -- so the
            # skip path carried nothing that had been computed, it simply injected
            # the voxel's identity into the output. Hence |q|=55.7 vs |attn_out|=1.11
            # mattering at all, which in a normal block it would not.
            #
            # Applying the SAME block repeatedly gives h something to accumulate. The
            # weights are shared, so this adds ZERO parameters: it isolates depth of
            # refinement from capacity, which stacking independent layers cannot do
            # (each would add ~2.36M and make the decoder larger than the 3.15M linear
            # readout it replaced -- the comparison the whole arm rests on).
            # Weight-tied repetition follows Universal Transformer / ALBERT; repeated
            # cross-attention to a fixed memory follows Perceiver.
            #
            # iters=1 is exactly the legacy single pass, so the default is untouched.
            h = q
            attn_w = None
            for i in range(self.iters):
                a, w = self.attn(
                    self.ln_q(h), m, m,
                    key_padding_mask=memory_key_padding_mask,
                    # only the last pass' map is the one the probe wants
                    need_weights=(need_weights and i == self.iters - 1))
                # The residual rule, applied per pass. Under 'raw' this is h = h + a
                # and h stays pinned to q (see the note in model.py); under 'balanced'
                # both sides sit at sqrt(D), so h is genuinely rewritten each pass and
                # the next query is a real mix of identity and evidence. 'evidence'
                # drops the skip entirely -- at depth that is the honest test of
                # whether the skip earns its place, which at depth 1 it could not be.
                if self.residual == 'raw':
                    h = h + a
                elif self.residual == 'normed':
                    h = self.ln_q(h) + a
                elif self.residual == 'balanced':
                    h = self.ln_q(h) + self.ln_a(a)
                else:                                    # 'evidence'
                    h = self.ln_a(a)
                if w is not None:
                    attn_w = w
                if self.use_ffn:
                    h = h + self.ffn(self.ln_f(h))
            logits = self.head(self.ln_o(h)).squeeze(-1).transpose(0, 1)
            return logits + self.bias(), attn_w

        q_normed = self.ln_q(q)
        attn_out, attn_w = self.attn(
            q_normed, m, m,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights)
        if self.residual == 'raw':
            h = q + attn_out                        # |q| ~ 55, |attn_out| ~ 1
        elif self.residual == 'normed':
            h = q_normed + attn_out                 # |q_normed| = sqrt(D) ~ 27.8
        elif self.residual == 'balanced':
            h = q_normed + self.ln_a(attn_out)      # both sides at sqrt(D)
        elif self.residual == 'evidence':
            # q reaches the output ONLY as the attention query -- it never touches
            # h. A voxel's logit is then a function of what its query retrieved,
            # not of which voxel it is, so the model cannot score a cell high on
            # identity alone. This is the arm that tests where the static map lives:
            # 91% of logit variance is static per-voxel on the P08 baseline, and if
            # that survives here it lives in the retrieved evidence (the kitchen
            # does not move) rather than in this residual.
            #
            # NOT the same as 'balanced' with the query turned down. normed (25:1)
            # and balanced (1:1) both left PR-AUC where raw had it on fold P01
            # (.0460 / .0495 / .0463), which is why this removes the term instead
            # of rescaling it a third time.
            h = self.ln_a(attn_out)
        else:  # 'concat'
            # Both sides survive, side by side rather than summed, and the head
            # gets 1536 inputs instead of 768. Nothing forces the model to drop q
            # -- but the two halves of head.weight now say how much it WANTED each,
            # which a sum cannot reveal (a sum makes them inseparable by
            # construction). Read those halves' norms off the trained checkpoint.
            h = torch.cat([q_normed, self.ln_a(attn_out)], dim=-1)
        if self.use_ffn:
            h = h + self.ffn(self.ln_f(h))
        logits = self.head(self.ln_o(h)).squeeze(-1).transpose(0, 1)  # [B, 4096]
        return logits + self.bias(), attn_w
