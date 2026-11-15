import argparse
import random
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments, set_seed

from boxer_env import MAX_OBB_TOKENS
from fixture_obbs import (DEFAULT_COS_THRESH as FIXTURE_DEFAULT_COS,
                          DEFAULT_IOU_THRESH as FIXTURE_DEFAULT_IOU)
from gaze import DEFAULT_TIME_OFFSET_S as GAZE_DEFAULT_TIME_OFFSET_S
from loader import HDEpicInteractionDataset, NUM_VOXELS, PARTICIPANT_TO_SPLIT
from model import CustomConfig, CustomTransformerModel
from object_class_vocab import NUM_OBJECT_CLASSES
from trainer import CustomTrainer
from trajectory_tokens import trajectory_raw_dim_simple


def str2bool(value):
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=float, default=0.1,
                        help='Eval + checkpoint cadence as a fraction of total training steps '
                             '(HF fractional steps), e.g. 0.05 with 5 epochs = every 0.25 epoch.')
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    parser.add_argument('--wandb_project', type=str, default='FIction-HD-EPIC-v2',
                        help="WandB project to log to. Default keeps every earlier run "
                             "in one place; use a separate project for a distinct line "
                             "of work (e.g. FIction-HD-EPIC-OBB_tokens).")
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_modality', type=str, default='video,location,env',
                        help="Comma-separated subset of {'video','location','env','obb',"
                             "'gaze','trajectory'} (order fixed). 'obb' is the per-box "
                             "alternative to 'env': one token per fused 3D OBB (~162/window "
                             "with fixtures, ~124 without) "
                             "instead of one per voxel (4096).")
    parser.add_argument('--max_obb_tokens', type=int, default=MAX_OBB_TOKENS,
                        help="Cap on OBB tokens per window for the 'obb' modality. Under "
                             "'iou' fusion the raw per-window count is mean 124 / max 276 "
                             "over all 28,777 cached windows; with --fixture_obbs it is "
                             "mean 162 / max 317. The default %(default)s truncates 1.9%% "
                             "of fixture-augmented windows, always weakest-detection first "
                             "(fixtures carry prob=1.0, above every Boxer prob).")
    parser.add_argument('--type_embeddings', type=str, default='separate',
                        choices=['separate', 'unified'],
                        help="Modality markers. 'separate' (legacy) is four independently "
                             "named parameters and NO marker for video. 'unified' is one "
                             "[5, H] table covering video too, added AFTER the segment "
                             "LayerNorm rather than inside it.")
    parser.add_argument('--type_embedding_scale', type=float, default=1.0,
                        help="std multiplier on the unified table's randn init. 1.0 gives "
                             "each marker norm sqrt(768)=27.7, matching a LayerNorm'd "
                             "token and the legacy markers. 0.02 would give 0.55, i.e. 2%% "
                             "of the token.")
    parser.add_argument('--class_embedding_gain', type=str2bool, default=False,
                        help="Give the projected class vectors a learned magnitude. Without "
                             "it they sit at 0.71 against 47.8/27.7/27.5 for the position, "
                             "volume and marker terms of the same OBB token, and training at "
                             "lr 1e-5 does not move them. --class_embedding_init only.")
    parser.add_argument('--class_embedding_gain_init', type=float, default=-1.0,
                        help="Init for that scalar; <0 means sqrt(hidden)=27.71, where the "
                             "learned rows and the volume/marker terms already sit.")
    parser.add_argument('--obb_voxel_grid_prune', type=str2bool, default=False,
                        help="Drop fused OBBs whose centre lies outside the participant's "
                             "voxel grid (1.71%% of all boxes). Without it boxer_env's "
                             "_nearest_voxel_ijk silently snaps them onto a boundary voxel. "
                             "Requires --voxel_centers_dir. Implied by --fixture_obbs.")
    parser.add_argument('--fixture_obbs', type=str2bool, default=False,
                        help="Add the participant's static Digital-Twin fixture OBBs "
                             "(fixture_obbs.py) and drop the fused OBBs they already cover. "
                             "Each sample uses only its OWN participant's fixtures.")
    parser.add_argument('--fixture_iou_thresh', type=float, default=FIXTURE_DEFAULT_IOU,
                        help="Clause (a) of the fixture filter: 3D IoU above this against "
                             "the SAME fixture that also passes the class gate.")
    parser.add_argument('--fixture_cos_thresh', type=float, default=FIXTURE_DEFAULT_COS,
                        help="Clause (b): OWLv2 class-embedding cosine above this. A class "
                             "with no embedding never matches, so it is never dropped.")
    parser.add_argument('--use_static_env', type=str2bool, default=True,
                        help='v2 uses a single Boxer env snapshot per observation window.')
    parser.add_argument('--env_source', type=str, default='boxer_grid',
                        choices=['boxer_grid', 'detic_grid'],
                        help="Which detector defines the voxel-grid extent. "
                             "'boxer_grid': AABB(Boxer OBBs + SLAM traj). "
                             "'detic_grid': AABB(Detic OBBs + SLAM traj); env "
                             "content is still Boxer.")
    parser.add_argument('--voxel_centers_dir', type=str, default='',
                        help="If set, use fixed per-participant voxel grids from "
                             "<dir>/<pid>_voxel_centers.npy instead of deriving the "
                             "grid per-video (env_source is then ignored; env "
                             "content stays Boxer).")
    parser.add_argument('--min_video_duration', type=float, default=0.0,
                        help="Skip videos shorter than this many seconds (from "
                             "Videos/<pid>/durations.txt) before building any "
                             "episodes for them. 0 = no filtering.")
    parser.add_argument('--require_valid_location', type=str2bool, default=False,
                        help="Drop episodes whose observation window has no valid "
                             "SLAM location second. Needed when 'location' is the "
                             "ONLY modality (an all-padding window masks the whole "
                             "sequence and attention returns NaN).")
    parser.add_argument('--gt_interactions_range', type=str, default='',
                        help="Keep only episodes whose prediction window holds a GT "
                             "INTERACTION count in this inclusive 'lo,hi' range (e.g. "
                             "'1,92'). Empty = no filter. Counts raw events, so several "
                             "may land in one voxel.")
    parser.add_argument('--gt_voxels_range', type=str, default='',
                        help="Keep only episodes whose prediction window holds a "
                             "DISTINCT GT VOXEL count in this inclusive 'lo,hi' range "
                             "(e.g. '3,40'). Empty = no filter. Counts what "
                             "labels['interaction'] actually contains, i.e. after "
                             "duplicates collapse into a voxel. An episode must pass "
                             "this AND --gt_interactions_range to survive. Applies to "
                             "every split; note it changes the positive rate, so the "
                             "resulting PR-AUC is not comparable with an unfiltered run "
                             "-- compare against a prior recomputed on the same subset.")
    # --- Ablation toggles (each independent; see boxer_env.py/model.py/trajectory_tokens.py) ---
    parser.add_argument('--env_orphan_fallback', type=str2bool, default=True,
                        help="Assign Boxer boxes too small/off-centre to contain any voxel "
                             "centre to their nearest voxel (default). False = traditional "
                             "method: such boxes are simply dropped.")
    parser.add_argument('--fusion', type=str, default='grid', choices=['grid', 'iou'],
                        help="How per-window Boxer detections are clustered into fused "
                             "boxes, feeding BOTH the env grid and the OBB tokens. "
                             "'grid' (default, legacy) snaps centres to a 0.5m cell and "
                             "groups by (name, cell). 'iou' links detections that actually "
                             "overlap in 3D and whose class names are close in OWLv2 space, "
                             "then chains the links (see iou_fusion.py) -- it costs ~6.4s "
                             "per window against ~0.04s, so it REQUIRES --fused_cache.")
    parser.add_argument('--fused_cache', type=str2bool, default=False,
                        help="Read precomputed fused boxes from "
                             "boxer_obbs/<video_id>/fused_obbs_iou.pkl, written by "
                             "precompute_fused_obbs.py. Required in practice by "
                             "--fusion iou. Each file records the settings it was built "
                             "under (including t0, which --time_anchor sets) and the "
                             "loader raises on a mismatch, since a wrong cache has the "
                             "same window keys holding different boxes.")
    parser.add_argument('--boxer_conf_threshold', type=float, default=-1.0,
                        help="Minimum Boxer detection probability before fusion. -1 (the "
                             "default) picks per strategy: 0.5 for --fusion grid, which is "
                             "what every earlier run used, and 0.6 for --fusion iou.")
    parser.add_argument('--trajectory_simple', type=str2bool, default=False,
                        help="Simplified trajectory tokens: start/end times + voxel "
                             "coordinates only (embedded via the shared location tables), "
                             "no interior shape points, no person-path branch check.")
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help="Forked DataLoader workers. 4 is what every run has used; "
                             "set 0 to run the documented local smoke test on a login "
                             "node, which caps threads and cannot fork workers.")
    parser.add_argument('--trajectory_token', type=str, default='mapper',
                        choices=['mapper', 'end_distance'],
                        help="Composition of a trajectory token. 'mapper' (legacy) sums "
                             "trajectory_mapper(raw features) + class + both endpoint "
                             "positions. 'end_distance' drops the MLP for "
                             "class + pos(end voxel) + distance bin + trajectory_type, "
                             "so a track is 'a <class> ended up here having moved this "
                             "far'. Requires --trajectory_simple true. Carries NO time "
                             "term: a movement 2s ago and one 28s ago are one token.")
    parser.add_argument('--max_trajectory_tokens', type=int, default=16,
                        help="Cap on trajectory tokens per episode, keeping the tracks "
                             "with the most recent observed end. 16 is the value every "
                             "earlier trajectory run used (it was hardcoded in loader.py "
                             "until this flag existed), so leave it alone to stay "
                             "comparable with them.")
    parser.add_argument('--class_embedding_init', type=str, default='',
                        help="Path to class_embeddings.npz (OWLv2 text vectors, the same "
                             "file --fusion iou uses) to seed class_embedding with. Empty "
                             "(default) keeps the legacy randomly-initialised table. "
                             "Applies to every modality that names an object -- env, obb "
                             "AND trajectory -- since they share one table.")
    parser.add_argument('--class_embedding_trainable', type=str2bool, default=False,
                        help="Under --class_embedding_init, whether the 1220 semantic rows "
                             "themselves train. Default False freezes them and learns only "
                             "the 512->768 projection, so fine-tuning on six kitchens "
                             "cannot pull apart classes the held-out kitchen needs.")
    parser.add_argument('--trajectory_exclude_unfinished', type=str2bool, default=False,
                        help="Drop object tracks still in motion at the end of the "
                             "observation window instead of clipping them to it. "
                             "Clipping interpolates along the straight line to the "
                             "track's TRUE end, which lies in the future, so the clipped "
                             "token leaks the destination's direction and arrival speed -- "
                             "and the track may be one of that episode's own GT events. "
                             "Default False preserves the earlier runs' behaviour.")
    parser.add_argument('--env_positional_encoding', type=str2bool, default=False,
                        help="Add each voxel's (ix,iy,iz) embedding (shared location tables) "
                             "to its env token, so the model can tell neighbouring voxels apart.")
    parser.add_argument('--env_class_averaging', type=str2bool, default=False,
                        help="Average the class embeddings of every object overlapping a voxel "
                             "(up to --env_max_classes_per_voxel) instead of keeping only the "
                             "highest-confidence winner.")
    parser.add_argument('--env_max_classes_per_voxel', type=int, default=4,
                        help="K: max number of classes averaged per voxel when "
                             "--env_class_averaging or --env_fallback_averaging is set.")
    parser.add_argument('--env_fallback_averaging', type=str, default='off',
                        choices=['off', 'merge', 'empty_only'],
                        help="Fill the grid with one winning class per voxel first, then "
                             "average in ONLY the boxes that pass missed (those containing "
                             "no voxel centre), assigning each to its nearest voxel. "
                             "'merge': keep the missed box's class alongside whatever won "
                             "that voxel. 'empty_only': keep it only where the voxel is "
                             "still empty. 'off': disabled.")
    parser.add_argument('--gaze_max_hz', type=float, default=0.0,
                        help="Thin gaze samples to at most this rate (0 = keep all, the "
                             "default). P01-P03 record at 10 Hz and P04-P09 at 60 Hz; more "
                             "samples is a strictly better estimate of the densest voxel, "
                             "so this exists only to test the two rates against each other.")
    parser.add_argument('--gaze_time_offset_s', type=float, default=GAZE_DEFAULT_TIME_OFFSET_S,
                        help="Residual offset added to the gaze time anchor (video second 0 "
                             "== first gaze timestamp + this). See gaze.py's docstring for "
                             "the measurement; 0 disables it.")
    parser.add_argument('--time_anchor', type=str, default='trajectory',
                        choices=['trajectory', 'gaze'],
                        help="Which device-clock instant is video second 0 for the "
                             "device-clock inputs (person_locations + Boxer env windows). "
                             "'trajectory' (legacy) uses the SLAM trajectory's first "
                             "sample, which starts ~0.95-1.50s into the recording, so "
                             "those inputs are read ~1s later in the video than the "
                             "window asks for. 'gaze' uses the validated anchor gaze.py "
                             "already uses (first gaze timestamp + gaze_time_offset_s), "
                             "putting location, env, gaze, video and labels on one clock.")
    # --- Readout / decoder (see docs/voxel_query_decoder_spec.md) ---
    parser.add_argument('--readout', type=str, default='token0',
                        choices=['token0', 'cls'],
                        help="Which sequence position is decoded to the 4096 voxel "
                             "logits. 'token0' (legacy) reads seat 0, which is always "
                             "the first visual token -- a content token doubling as the "
                             "global summary. 'cls' prepends a dedicated learned token "
                             "instead. Ignored by --decoder voxel_query for the readout "
                             "itself (the queries do that), but the CLS token still "
                             "joins the sequence and is attendable as memory.")
    parser.add_argument('--voxel_bias_init', type=str, default='zero',
                        choices=['zero', 'prior', 'empirical', 'empirical_factorised'],
                        help="Initial value of the output bias. 'zero' (legacy) keeps "
                             "nn.Linear's own init. 'prior' sets every entry to "
                             "log(p/(1-p)) for the TRAIN SPLIT's measured positive rate "
                             "-- the standard fix for the ~1:200 class imbalance, which "
                             "is otherwise unaddressed (BCEWithLogitsLoss, no pos_weight). "
                             "A scalar applied uniformly. The rate is measured and logged "
                             "whatever this says. "
                             "'empirical' instead sets each voxel to ITS OWN train-split "
                             "log-odds, and 'empirical_factorised' to the additive fit of "
                             "that field (axis marginals only). Both hand the model the "
                             "training kitchens' spatial layout on purpose -- see the "
                             "warning below; they are measurement arms, not defaults.")
    parser.add_argument('--voxel_bias_scale', type=float, default=1.0,
                        help="Scales the SPATIAL part of the bias field, around the pooled "
                             "prior that stays fixed: bias = pooled + scale*(field - pooled). "
                             "1.0 = the measured field, 0.0 collapses to --voxel_bias_init "
                             "prior, 2.0 exaggerates it, and -1.0 gives the anti-prior (the "
                             "field inverted about the pooled logit). No effect under "
                             "'zero'/'prior', which carry no spatial part.")
    parser.add_argument('--voxel_bias_lr', type=float, default=-1.0,
                        help="Learning rate for the voxel bias alone. -1 (default) leaves it "
                             "in the normal group at --learning_rate, where a scalar's total "
                             "travel is lr*T/2 = 0.097 at 1e-5 over a 19.4k-step run -- "
                             "measured drift over 27 finished voxel_query runs was at most "
                             "0.036, i.e. the bias is FROZEN AT ITS INIT in every result so "
                             "far. 0 freezes it explicitly (requires_grad=False), which is "
                             "the honest arm if the init is the thing under test. A positive "
                             "value gives it its own group, e.g. 1e-3 (trainer.FAST_LR) makes "
                             "it genuinely learnable for the first time.")
    parser.add_argument('--decoder', type=str, default='linear',
                        choices=['linear', 'voxel_query'],
                        help="'linear' (legacy) decodes one aggregated token through "
                             "Linear(768->4096): an independent 768-vector per voxel, "
                             "3.15M parameters, no way to route evidence about a region "
                             "to the voxels in it. 'voxel_query' makes each voxel's "
                             "coordinate a query over the sequence and shares ONE "
                             "Linear(768->1) across all 4096 (see decoder.py). Default "
                             "stays 'linear' so earlier runs remain reproducible; "
                             "production scripts opt in explicitly.")
    parser.add_argument('--voxel_query_type', type=str, default='axis_sum',
                        choices=['axis_sum', 'fourier', 'learned'],
                        help="How a voxel coordinate becomes a query (--decoder "
                             "voxel_query only). 'axis_sum': the shared "
                             "location_embedding_x/y/z tables + a query marker, keeping "
                             "the three axes separable. 'fourier': learnable Fourier "
                             "features, continuous but axis-mixing. 'learned': a private "
                             "4096x768 table -- the control arm, same parameter count as "
                             "the layer being removed and no coordinate structure.")
    parser.add_argument('--voxel_bias', type=str, default='factorised',
                        choices=['factorised', 'full', 'mlp'],
                        help="The decoder's unconditional per-voxel prior. 'factorised': "
                             "bias_x[i]+bias_y[j]+bias_z[k], 48 parameters -- expresses a "
                             "height prior, which transfers, without capacity to memorise "
                             "one kitchen's floor plan, which cannot (the grids are "
                             "wall-fitted but nothing fixes which wall is +x). 'full': a "
                             "[4096] vector, kept as an ablation arm. 'mlp': a small Tanh "
                             "network of (x,y,z) -- can express cross-axis structure "
                             "factorised cannot, while staying far short of full's per-voxel "
                             "memorisation capacity. Only meaningful with --voxel_bias_init "
                             "zero/prior and --voxel_bias_lr > 0 -- it has no closed-form fit "
                             "for a field, so it has to learn its shape rather than be "
                             "handed one; anything else raises.")
    parser.add_argument('--voxel_bias_mlp_hidden', type=str, default='32,32',
                        help="Comma-separated hidden widths for --voxel_bias mlp, e.g. "
                             "'32,32' (two hidden layers, ~1.2k params) or '16,16' (~350). "
                             "Ignored for 'factorised'/'full'.")
    parser.add_argument('--voxel_query_fourier_scale', type=str, default='10.0',
                        help="std of the Fourier features' W_r init "
                             "(--voxel_query_type fourier). Sensitive: too small and all "
                             "4096 queries collapse toward one vector, too large and "
                             "adjacent voxels alias. Ablate {2, 10, 30}. Accepts one "
                             "value, or three comma-separated ones for a per-axis init: "
                             "the grids are not cubic and not cubic alike -- measured over "
                             "new_voxel_centers/, the z cell is 13.3-14.6cm in every "
                             "kitchen while x runs 16.8-41.7cm and y 12.6-46.6cm, so one "
                             "scale resolves height far more finely than either floor axis.")
    parser.add_argument('--position_encoding', type=str, default='auto',
                        choices=['auto', 'none', 'fourier'],
                        help="Positional code on the INPUT tokens carrying a full (i,j,k) "
                             "-- obb, env, and trajectory endpoints. 'auto' (default) "
                             "follows --voxel_query_type, so 'fourier' there makes the "
                             "queries and the input tokens share ONE code instead of two "
                             "unrelated ones; with the default axis_sum it resolves to "
                             "'none' and nothing changes. 'none' forces the index tables "
                             "(and reproduces the 2026-08-11 vq_fourier runs, whose queries "
                             "had a private W_r). 'fourier' forces the code on. NOT applied "
                             "to location or gaze: those are fed as three separate per-axis "
                             "token streams, so a token there holds one scalar, not a "
                             "coordinate. 1152 + 769 parameters.")
    parser.add_argument('--position_encoding_mode', type=str, default='replace',
                        choices=['replace', 'add'],
                        help="'replace' (default): an input token's position IS the Fourier "
                             "code, the exact structural analogue of the axis_sum arm and "
                             "what makes a query and an input token differ by their type "
                             "marker alone. 'add': index tables + Fourier, keeping what a "
                             "lookup expresses and a Fourier basis cannot (non-smooth "
                             "structure, 'the counter runs along the k=8 slab') at the cost "
                             "of the queries no longer carrying the same vector.")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help="AdamW weight decay. 0.0 is HuggingFace's default and what "
                             "every run so far has used -- i.e. this model has never had "
                             "any L2 at all. 0.01-0.1 is standard for transformers and is "
                             "the cheapest single change available against a held-out "
                             "PR-AUC that peaks at step ~500 of ~19,400 and decays after.")
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help="HuggingFace TrainingArguments schedule name (linear, cosine, "
                             "constant_with_warmup, cosine_with_restarts, ...). 'linear' is "
                             "HF's default and what every run so far has used: LR decays "
                             "linearly to 0 over the whole run with no plateau. Applies to "
                             "every optimizer param group identically (including the "
                             "FAST_PARAMS group in trainer.py), since HF's LambdaLR scales "
                             "each group's own base lr by the same schedule multiplier.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                        help="Fraction of total steps spent ramping LR linearly from 0 to "
                             "its base value before the schedule above takes over. 0.0 is "
                             "HF's default and what every run so far has used -- LR starts "
                             "at ~100%% of base on step 1.")
    parser.add_argument('--encoder_dropout', type=float, default=0.1,
                        help="Dropout inside each TransformerEncoderLayer. 0.1 is "
                             "PyTorch's default, which every earlier run silently used, "
                             "so this changes nothing until you raise it.")
    parser.add_argument('--decoder_dropout', type=float, default=0.0,
                        help="Dropout in the voxel-query decoder's cross-attention. The "
                             "default 0.0 is nn.MultiheadAttention's, so the 4096 queries "
                             "have attended with no dropout at all while the encoder ran "
                             "at 0.1 -- an asymmetry nothing chose.")
    parser.add_argument('--obb_token_dropout', type=float, default=0.0,
                        help="Train-time probability of hiding each OBB token. The "
                             "TARGETED regulariser: if the model overfits by identifying "
                             "the training kitchen from its object inventory, hiding a "
                             "random share of the ~250 boxes each step makes any single "
                             "identifying configuration unreliable, while the part that "
                             "generalises (a hob exists, at this height) survives. Weight "
                             "decay cannot express that. Off at 0.0; try 0.1-0.3.")
    parser.add_argument('--location_tokens', type=str, default='axis',
                        choices=['axis', 'joint'],
                        help="Layout of the 'location' segment. 'axis' (legacy): three "
                             "tokens per second, one per axis, 90 tokens -- a location "
                             "token holds a SCALAR and the axes combine only downstream "
                             "via attention, relying on the shared time stamp to say "
                             "'these three are one instant' (and that stamp is 2.6%% of "
                             "the token under --time_encoding_gain fixed). 'joint': one "
                             "token per second carrying the whole position through the "
                             "same _voxel_pos_embedding every other positional token "
                             "uses, so it inherits --position_encoding; 30 tokens, +768 "
                             "parameters for the type marker. Not strictly better: 'axis' "
                             "gives attention three independent slots, so a query for a "
                             "high voxel can weight loc_z and ignore floor position, "
                             "which 'joint' cannot do.")
    parser.add_argument('--loss', choices=['bce', 'focal'], default='bce',
                        help="'focal' adds Lin et al. (ICCV 2017) focal loss. The repo already "
                             "uses that paper's bias init via --voxel_bias_init prior; this is "
                             "the loss reshaping it was designed to accompany.")
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focusing parameter; 0 reduces focal exactly to BCE')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help="positive-class weight (paper value 0.25); -1 disables")
    parser.add_argument('--pos_weight', type=float, default=-1.0,
                        help='>0 weights the positive class in plain BCE -- the simpler, '
                             'older fix and the honest control for --loss focal')
    parser.add_argument('--decoder_iters', type=int, default=1,
                        help='how many times the ONE cross-attention block is applied, weights '
                             'shared (needs --decoder_residual raw). 1 = legacy single pass; '
                             '>1 gives the residual an accumulating h at zero extra parameters')
    parser.add_argument('--voxel_query_untied', type=str2bool, default=False,
                        help="voxel_query_type='fourier' only: give the decoder its own "
                             "W_r/gain/marker instead of sharing the model's, so tied-vs-"
                             "untied is a one-variable comparison (PETR's arrangement "
                             "against DAB-DETR's). Default off, creates no parameter.")
    parser.add_argument('--position_encoding_gain_init', type=float, default=48.0,
                        help="Init for the scalar setting the spatial code's magnitude. "
                             "cos/sin are bounded, so W_r sets frequencies and can NEVER "
                             "set amplitude -- the decoder's legacy /sqrt(D) pinned the "
                             "code at 0.707, which is 2.5%% of an OBB token's class "
                             "embedding (27.7) and turns --decoder_residual raw from a 50:1 "
                             "regime into 0.6:1. sqrt(3*768)=48.0 is the measured norm of "
                             "the three summed index rows it replaces, so the fourier and "
                             "axis_sum arms start matched. Like time_gain it gets its own "
                             "lr=1e-3 group in trainer.py (a scalar cannot travel at 1e-5), "
                             "so this is an INIT, not a setting -- read the final value out "
                             "of the checkpoint.")
    parser.add_argument('--gaze_tokens', type=str, default='axis',
                        choices=['axis', 'joint'],
                        help="Layout of the gaze stream, the exact analogue of "
                             "--location_tokens. 'axis' (legacy) feeds three 30-token "
                             "streams, one per spatial axis, each token holding a SCALAR "
                             "-- which is WHY gaze sat outside --position_encoding's "
                             "scope: a 3-D code has nowhere to go on a one-number token, "
                             "so gaze kept the index tables while obb, location-joint and "
                             "the decoder queries moved to the Fourier code, leaving a "
                             "third spatial vocabulary in the model. 'joint' feeds one "
                             "token per second carrying the looked-at voxel through "
                             "_voxel_pos_embedding, 90 tokens -> 30, so gaze finally "
                             "shares whatever --position_encoding selects. Keeps "
                             "gaze_type_embedding either way: under 'joint' a bare gaze "
                             "token would otherwise be byte-identical to a joint LOCATION "
                             "token at the same voxel, and to a decoder query there.")
    parser.add_argument('--decoder_heads', type=int, default=8,
                        help="Attention heads in the voxel-query decoder.")
    parser.add_argument('--decoder_ffn', type=str2bool, default=False,
                        help="Add an FFN sublayer after the decoder's cross-attention. "
                             "Off by default: +4.72M parameters makes the model LARGER "
                             "than the readout it replaces, which undercuts the point. "
                             "Report the parameter count with the metric if you enable it.")
    parser.add_argument('--decoder_ffn_dim', type=int, default=3072,
                        help="Width of that FFN when --decoder_ffn is set.")
    parser.add_argument('--time_encoding', type=str, default='none',
                        choices=['none', 'fourier', 'sinusoidal', 'lookup'],
                        help="Temporal encoding on the per-second tokens (video, "
                             "location, gaze). 'none' (legacy) leaves the 30 seconds an "
                             "unordered SET to self-attention -- permuting them does not "
                             "change the output, so recency is unreadable. 'fourier' adds "
                             "a learned Fourier encoding of time-to-anchor "
                             "(delta = (T-1-n)/T in [0,1), 0 = the second nearest the "
                             "anchor). Not applied to obb (a fused box has no timestamp), "
                             "env (not per-second), or trajectory (already carries "
                             "sinusoidal endpoint times). 384 parameters. 'lookup' is "
                             "MMT's scheme (arXiv:2007.10639): one learned vector per "
                             "slot, 30x768=23,040 parameters, binned by seconds-before-"
                             "anchor rather than MMT's absolute video second. It needs "
                             "no --time_encoding_gain: nn.Embedding inits N(0,1) so a "
                             "row sits at sqrt(768)=27.7, against the 0.707 that "
                             "fourier+gain=fixed produces. It buys that amplitude by "
                             "giving up smoothness -- 30 unrelated vectors, nothing "
                             "tying slot 5 to slot 6. Measure the trade with "
                             "analysis/time_code_similarity.py before choosing.")
    parser.add_argument('--time_encoding_max_wavelength', type=float, default=0.0,
                        help="--time_encoding sinusoidal only. Longest wavelength of "
                             "the geometric ladder, in slots. 0 (default) means "
                             "2*observation_time, NOT the literature's 10000: that "
                             "figure is tuned for sequences of thousands and over 30 "
                             "slots most of the ladder is constant -- measured cosine "
                             "never falls below 0.567 and the dynamic range collapses "
                             "to 0.406, against 0.898 at 2T. Set it explicitly only "
                             "to reproduce an old run or to sweep the horizon.")
    parser.add_argument('--time_encoding_scale', type=float, default=10.0,
                        help="std of the time encoding's W_r init. NOT a range for delta, "
                             "which is always [0,1): this sets how fast the features "
                             "oscillate across the window. Too small and all 30 seconds "
                             "collapse to one vector; too large and adjacent seconds "
                             "alias. Ablate {2, 10, 30}.")
    parser.add_argument('--time_encoding_gain', type=str, default='fixed',
                        choices=['fixed', 'learned'],
                        help="Magnitude of the time code. 'fixed' (legacy) divides "
                             "[cos,sin] by sqrt(D) for a norm of 0.707 -- measured on a "
                             "trained te10 checkpoint that is 2.6%% of a location token "
                             "(27.7) and 12%% of a visual token (5.9), and W_r cannot fix "
                             "it: cos/sin are bounded, so W_r sets frequencies and never "
                             "amplitude (it moved 0.025%% in 19k steps). 'learned' "
                             "unit-norms the code and multiplies by a trainable scalar, "
                             "giving the model a magnitude to choose. 1 parameter.")
    parser.add_argument('--time_encoding_gain_init', type=float, default=8.0,
                        help="Init for that scalar. Against --token_layernorm tokens "
                             "(norm sqrt(768)=27.7) 8.0 is a 29%% contribution. It is an "
                             "INIT, not a setting: trainer.py puts time_gain in its own "
                             "lr=1e-3 group (~9.7 of travel over a 19.4k-step run) "
                             "precisely so the run measures the amplitude instead of "
                             "assuming it. Read the final value out of the checkpoint.")
    parser.add_argument('--token_layernorm', type=str2bool, default=False,
                        help="LayerNorm each segment's tokens before concatenation. "
                             "Segment scale is currently an accident of init convention: "
                             "nn.Embedding is N(0,1) so location lands at 27.7 and obb at "
                             "66.9, while nn.Linear's fan_in=4096 leaves video at 5.9 and "
                             "cls_token (std 0.02) at 0.56. nn.TransformerEncoderLayer is "
                             "post-LN, so layer 1 sees exactly those raw magnitudes. One "
                             "LayerNorm per segment TYPE, 12,288 parameters.")
    parser.add_argument('--decoder_residual', type=str, default='raw',
                        choices=['raw', 'normed', 'balanced', 'evidence', 'concat'],
                        help="How the voxel query re-enters the stream after "
                             "cross-attention. 'raw' (legacy) is h = q + attn_out; on a "
                             "trained axis_sum checkpoint |q|=55.7 vs |attn_out|=1.11, so "
                             "the vector the head reads is 50:1 the query and evidence is "
                             "a small perturbation on it. 'normed' is ln_q(q) + attn_out "
                             "(25:1 -- LayerNorm output is still sqrt(D)). 'balanced' is "
                             "ln_q(q) + ln_a(attn_out), both sides at sqrt(D), the arm "
                             "that actually equalises them. "
                             "Those three all KEEP the query in the output and only change "
                             "its magnitude, and on fold P01 they land together (.0463 / "
                             ".0460 / .0495) -- so the two below remove it instead. "
                             "'evidence' is h = ln_a(attn_out): the query still selects what "
                             "to attend to but never reaches the head, so a voxel cannot "
                             "score on identity alone. 'concat' is [ln_q(q); ln_a(attn_out)] "
                             "through a 1536->1 head, which keeps both but lets the head "
                             "weight them separately -- the two halves of head.weight then "
                             "read out how much the model wanted each. 'concat' is "
                             "incompatible with --decoder_ffn (different widths). "
                             "See decoder.py's docstring.")
    # --- Runtime episode hyperparameters (the whole point of v2) ---
    parser.add_argument('--observation_time', type=int, default=30)
    parser.add_argument('--anticipation_time', type=float, default=5)
    parser.add_argument('--future_time', type=float, default=180)
    # --- Splits (participant-level) ---
    parser.add_argument('--train_participants', type=str, default='P01,P02,P03,P04,P05,P06')
    parser.add_argument('--val_participants', type=str, default='P07')
    parser.add_argument('--test_participants', type=str, default='P08,P09')
    parser.add_argument('--train_n', type=int, default=0, help='Cap on train episodes (0 = all).')
    parser.add_argument('--val_n', type=int, default=0, help='Cap on val episodes (0 = all).')
    parser.add_argument('--test_n', type=int, default=0, help='Cap on test episodes (0 = all).')
    parser.add_argument('--debug_subset', type=int, default=0,
                        help='If > 0, truncate every split to this many samples for debugging.')
    return parser.parse_args()


def torch_default_data_collator(features):
    """Merge a list of per-sample dicts into one batched dict (nested 'labels' handled)."""
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is None or isinstance(v, str):
            continue
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        elif isinstance(v, np.ndarray):
            batch[k] = torch.tensor(np.stack([f[k] for f in features]))
        elif isinstance(v, dict):
            batch[k] = {}
            for sub_key in v:
                sub_vals = [f[k][sub_key] for f in features]
                if isinstance(sub_vals[0], np.ndarray):
                    batch[k][sub_key] = torch.tensor(np.stack(sub_vals))
                else:
                    batch[k][sub_key] = torch.stack(sub_vals)
        else:
            batch[k] = torch.tensor([f[k] for f in features])
    return batch


class ComputeMetricsWrapper:
    """Metric logic passed to HuggingFace Trainer (object-only interaction task)."""

    def ensure_top_k_ones(self, predictions, threshold=0.1, top_k=3):
        binary = (predictions > threshold).astype(int)
        for i in range(predictions.shape[0]):
            if binary[i].sum() < top_k:
                top_k_idx = np.argsort(predictions[i])[-top_k:]
                binary[i, top_k_idx] = 1
        return binary

    def chamfer_distance(self, point_set_1, point_set_2, grid_size=16, normalize=True):
        from scipy.spatial import cKDTree
        if len(point_set_1) == 0 or len(point_set_2) == 0:
            return 0.0 if len(point_set_1) == 0 and len(point_set_2) == 0 else 1.0
        tree_1 = cKDTree(point_set_1)
        tree_2 = cKDTree(point_set_2)
        d1, _ = tree_1.query(point_set_2, k=1)
        d2, _ = tree_2.query(point_set_1, k=1)
        cd = np.mean(d1 ** 2) + np.mean(d2 ** 2)
        if normalize:
            d_max = np.sqrt(3 * (grid_size - 1) ** 2)
            cd = cd / (2 * d_max ** 2)
        return cd

    def process_and_calculate_chamfer(self, labels, binary_preds):
        labels_3d = labels.reshape(-1, 16, 16, 16)
        preds_3d = binary_preds.reshape(-1, 16, 16, 16)
        cds = []
        for i in range(labels_3d.shape[0]):
            label_pts = np.argwhere(labels_3d[i] == 1)
            pred_pts = np.argwhere(preds_3d[i] == 1)
            cds.append(self.chamfer_distance(label_pts, pred_pts))
        return np.mean(cds), cds

    def compute_metrics(self, eval_pred):
        predictions, labels, indexes = eval_pred
        interaction_logits = predictions[0] if isinstance(predictions, tuple) else predictions
        interaction_labels = labels['interaction']
        preds_sigmoid = 1 / (1 + np.exp(-interaction_logits))

        aucs_roc, aucs_pr = [], []
        for i in range(preds_sigmoid.shape[0]):
            y_true = interaction_labels[i]
            y_pred = preds_sigmoid[i]
            if len(np.unique(y_true)) > 1:
                aucs_roc.append(roc_auc_score(y_true, y_pred))
                aucs_pr.append(average_precision_score(y_true, y_pred))
        mean_roc_auc = np.nanmean(aucs_roc) if aucs_roc else 0.0
        mean_pr_auc = np.nanmean(aucs_pr) if aucs_pr else 0.0

        metrics = {}
        thresh_params = [
            {'thresh': 0.01, 'top_k': 5}, {'thresh': 0.05, 'top_k': 5},
            {'thresh': 0.1, 'top_k': 5}, {'thresh': 0.2, 'top_k': 5},
        ]
        for tp in thresh_params:
            binary_preds = self.ensure_top_k_ones(preds_sigmoid, threshold=tp['thresh'], top_k=tp['top_k'])
            desc = f"thresh-{tp['thresh']}-topk-{tp['top_k']}"
            avg_cd, _ = self.process_and_calculate_chamfer(interaction_labels, binary_preds)
            tol = 1e-10
            intersection = np.logical_and(binary_preds, interaction_labels).sum(axis=1)
            union = np.logical_or(binary_preds, interaction_labels).sum(axis=1)
            mean_iou = np.mean((intersection + tol) / (union + tol))
            metrics[f"chamfer_dist_{desc}"] = avg_cd
            metrics[f"iou_{desc}"] = mean_iou

        metrics['best_chamfer_dist'] = min(v for k, v in metrics.items() if k.startswith('chamfer_dist_'))
        metrics['roc_auc'] = mean_roc_auc
        metrics['pr_auc'] = mean_pr_auc
        metrics['indexes'] = indexes
        return metrics


def _base_and_indices(ds):
    """Unwrap nested Subsets (--train_n, --debug_subset) to the underlying
    HDEpicInteractionDataset and the episode indices actually in use."""
    idx = list(range(len(ds)))
    while isinstance(ds, Subset):
        idx = [ds.indices[i] for i in idx]
        ds = ds.dataset
    return ds, idx


def _nearest_voxel_flat(points, voxel_centers_flat, chunk=512):
    """Flat voxel index of the nearest grid centre for each point.

    The same nearest-centre rule loader.coord_to_voxel applies per point, done
    in blocks so a whole video's GT costs one pass instead of one per point.
    The flat index is C order over (16,16,16), i.e. i*256 + j*16 + k -- the
    layout labels['interaction'] itself uses.
    """
    out = np.empty(len(points), dtype=np.int64)
    for s in range(0, len(points), chunk):
        block = points[s:s + chunk]
        d = ((block[:, None, :] - voxel_centers_flat[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = d.argmin(axis=1)
    return out


def positive_rate_report(dataset, split='train'):
    """Fraction of the 4096 voxels labelled 1, averaged over episodes.

    Computed from each episode's gt_locations + its participant's voxel grid --
    the same two inputs __getitem__ uses to build labels['interaction'] -- so it
    costs no feature loading. Returns (pooled_rate, per_participant, n_episodes)
    where per_participant maps pid -> (rate, mean_count, n_episodes).

    Reported per participant as well as pooled because voxel size differs per
    kitchen, so the rate does too, and PR-AUC is prevalence-sensitive: the
    per-fold rates are what make per-fold scores commensurable.

    Also returns ``voxel_counts`` [4096]: how many episodes label each voxel
    positive, in the label flattening order. Costs one accumulate on indices the
    loop already computes. That vector is the TRAIN SPLIT'S SPATIAL LAYOUT, so it
    is only ever an input to --voxel_bias_init empirical/empirical_factorised,
    which exist to measure what handing the model that layout does -- see the
    warning on those flags.
    """
    base, indices = _base_and_indices(dataset)
    episodes = [base.dataset[i] for i in indices]

    by_video = {}
    for pos, ep in enumerate(episodes):
        by_video.setdefault(ep['video_id'], []).append(pos)

    counts = np.zeros(len(episodes), dtype=np.float64)
    voxel_counts = np.zeros(NUM_VOXELS, dtype=np.float64)
    for video_id, positions in by_video.items():
        vc_flat = base._voxel_centers_flat[video_id]
        pts, bounds = [], [0]
        for pos in positions:
            locs = np.asarray(episodes[pos]['gt_locations'], dtype=np.float64).reshape(-1, 3)
            pts.append(locs)
            bounds.append(bounds[-1] + len(locs))
        flat = _nearest_voxel_flat(np.concatenate(pts) if pts else np.empty((0, 3)),
                                   vc_flat)
        for n, pos in enumerate(positions):
            # Distinct voxels, not points: several GT objects can share a cell,
            # and the label grid is binary.
            u = np.unique(flat[bounds[n]:bounds[n + 1]])
            counts[pos] = len(u)
            voxel_counts[u] += 1

    per_participant = {}
    pids = np.array([ep['participant_id'] for ep in episodes])
    for pid in sorted(set(pids.tolist())):
        sel = counts[pids == pid]
        per_participant[pid] = (float((sel / NUM_VOXELS).mean()), float(sel.mean()), len(sel))
    pooled = float((counts / NUM_VOXELS).mean()) if len(counts) else float('nan')

    print(f"[{split}] positive rate: {pooled:.6f} of {NUM_VOXELS} voxels "
          f"({counts.mean():.1f} positive voxels per episode) over {len(counts)} episodes "
          f"-> prior logit {prior_logit(pooled):+.4f}")
    for pid, (rate, count, n) in per_participant.items():
        print(f"[{split}]   {pid}: {rate:.6f} ({count:.1f}/episode, n={n}, "
              f"logit {prior_logit(rate):+.4f})")
    return pooled, per_participant, len(counts), voxel_counts / max(len(counts), 1)


def prior_logit(rate):
    """log(p / (1-p)), clamped away from the degenerate ends."""
    p = min(max(float(rate), 1e-8), 1 - 1e-8)
    return float(np.log(p / (1 - p)))


def _cap(ds, n, rng):
    if n <= 0 or n >= len(ds):
        return ds
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return Subset(ds, idx[:n])


if __name__ == "__main__":
    args = get_args()
    if args.voxel_bias == 'mlp' and args.voxel_bias_lr <= 0:
        raise ValueError(
            "--voxel_bias mlp with --voxel_bias_lr <= 0 never leaves its flat "
            "init -- that is just --voxel_bias_init prior plus a pile of "
            "permanently-untrained weights. Set --voxel_bias_lr > 0.")
    set_seed(args.seed)
    rng = random.Random(args.seed)

    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    expt_name = f"{current_time}_{args.expt_name}" if '2024' not in args.expt_name else args.expt_name

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=expt_name, config=vars(args))

    config = CustomConfig(
        visual_feature_dim=4096,
        transformer_hidden_size=768,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=8,
        output_dim=4096,
        num_object_classes=NUM_OBJECT_CLASSES,  # boxer sem_id+1 (LVIS) + trajectory-only OOV slot
        observation_time=args.observation_time,
        use_static_env=args.use_static_env,
        input_modality=args.input_modality,
        trajectory_raw_dim=trajectory_raw_dim_simple() if args.trajectory_simple else None,
        trajectory_simple=args.trajectory_simple,
        trajectory_token=args.trajectory_token,
        class_embedding_init=args.class_embedding_init or None,
        class_embedding_trainable=args.class_embedding_trainable,
        env_positional_encoding=args.env_positional_encoding,
        # Both env-averaging modes hand the model the same [4096, K] class grid +
        # validity mask, so the model's averaging branch covers both; only the
        # loader distinguishes how those K classes were chosen.
        env_class_averaging=args.env_class_averaging or args.env_fallback_averaging != 'off',
        env_max_classes_per_voxel=args.env_max_classes_per_voxel,
        max_obb_tokens=args.max_obb_tokens,
        readout=args.readout,
        voxel_bias_init=args.voxel_bias_init,
        decoder=args.decoder,
        voxel_query_type=args.voxel_query_type,
        voxel_bias=args.voxel_bias,
        voxel_bias_mlp_hidden=args.voxel_bias_mlp_hidden,
        voxel_query_fourier_scale=args.voxel_query_fourier_scale,
        position_encoding=args.position_encoding,
        position_encoding_mode=args.position_encoding_mode,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        pos_weight=args.pos_weight,
        decoder_iters=args.decoder_iters,
        voxel_query_untied=args.voxel_query_untied,
        position_encoding_gain_init=args.position_encoding_gain_init,
        location_tokens=args.location_tokens,
        gaze_tokens=args.gaze_tokens,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        obb_token_dropout=args.obb_token_dropout,
        decoder_heads=args.decoder_heads,
        decoder_ffn=args.decoder_ffn,
        decoder_ffn_dim=args.decoder_ffn_dim,
        decoder_residual=args.decoder_residual,
        time_encoding=args.time_encoding,
        time_encoding_scale=args.time_encoding_scale,
        time_encoding_max_wavelength=args.time_encoding_max_wavelength,
        time_encoding_gain=args.time_encoding_gain,
        time_encoding_gain_init=args.time_encoding_gain_init,
        token_layernorm=args.token_layernorm,
        type_embeddings=args.type_embeddings,
        type_embedding_scale=args.type_embedding_scale,
        class_embedding_gain=args.class_embedding_gain,
        class_embedding_gain_init=(None if args.class_embedding_gain_init < 0
                                   else args.class_embedding_gain_init),
    )
    model = CustomTransformerModel(config)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    ds_kwargs = dict(observation_time=args.observation_time,
                     anticipation_time=args.anticipation_time,
                     future_time=args.future_time,
                     static_env=args.use_static_env,
                     env_source=args.env_source,
                     voxel_centers_dir=args.voxel_centers_dir or None,
                     min_video_duration=args.min_video_duration,
                     require_valid_location=args.require_valid_location,
                     gt_interactions_range=args.gt_interactions_range,
                     gt_voxels_range=args.gt_voxels_range,
                     env_orphan_fallback=args.env_orphan_fallback,
                     trajectory_simple=args.trajectory_simple,
                     trajectory_exclude_unfinished=args.trajectory_exclude_unfinished,
                     max_trajectory_tokens=args.max_trajectory_tokens,
                     env_class_averaging=args.env_class_averaging,
                     env_max_classes_per_voxel=args.env_max_classes_per_voxel,
                     env_fallback_averaging=args.env_fallback_averaging,
                     max_obb_tokens=args.max_obb_tokens,
                     gaze_max_hz=args.gaze_max_hz,
                     gaze_time_offset_s=args.gaze_time_offset_s,
                     time_anchor=args.time_anchor,
                     fusion=args.fusion,
                     fused_cache=args.fused_cache,
                     obb_voxel_grid_prune=args.obb_voxel_grid_prune,
                     fixture_obbs=args.fixture_obbs,
                     fixture_iou_thresh=args.fixture_iou_thresh,
                     fixture_cos_thresh=args.fixture_cos_thresh,
                     boxer_conf_threshold=(None if args.boxer_conf_threshold < 0
                                           else args.boxer_conf_threshold))
    val_pids = [p for p in args.val_participants.split(',') if p.strip()]
    no_val = len(val_pids) == 0   # train with no validation split -> select on test

    train_dataset = HDEpicInteractionDataset('train', participants=args.train_participants.split(','), **ds_kwargs)
    test_dataset = HDEpicInteractionDataset('test', participants=args.test_participants.split(','), **ds_kwargs)

    train_dataset = _cap(train_dataset, args.train_n, rng)
    test_dataset = _cap(test_dataset, args.test_n, rng)

    if no_val:
        # No validation participants: the test set drives periodic eval and best-
        # model selection (test thus influences selection; not a clean held-out test).
        print("[no-val] --val_participants empty: using the TEST set for eval + best-model selection.")
        eval_dataset = test_dataset
    else:
        eval_dataset = HDEpicInteractionDataset('val', participants=val_pids, **ds_kwargs)
        eval_dataset = _cap(eval_dataset, args.val_n, rng)

    if args.debug_subset > 0:
        train_dataset = _cap(train_dataset, args.debug_subset, rng)
        test_dataset = _cap(test_dataset, max(2, args.debug_subset // 5), rng)
        eval_dataset = test_dataset if no_val else _cap(eval_dataset, max(2, args.debug_subset // 5), rng)

    print(f"Train: {len(train_dataset)}  Val: {0 if no_val else len(eval_dataset)}  Test: {len(test_dataset)}")

    # --- Class prior (measured, never hardcoded) ---
    # Measured on every run, whatever --voxel_bias_init says: the per-participant
    # rates are needed to read per-fold PR-AUC (prevalence-sensitive) and cost one
    # pass over gt_locations. Only the bias init is conditional.
    (pooled_rate, per_participant_rate,
     n_train_episodes, voxel_rate) = positive_rate_report(train_dataset, 'train')
    config.voxel_prior_logit = prior_logit(pooled_rate)   # saved with the model config
    config.voxel_positive_rate = pooled_rate

    # The field handed to the bias. 'zero'/'prior' carry no spatial part, so they
    # stay scalars and the pre-existing path is untouched. The spatial arms build
    # [4096] log-odds from the TRAIN split only -- voxel_rate never sees val/test.
    bias_field = config.voxel_prior_logit
    if args.voxel_bias_init in ('empirical', 'empirical_factorised'):
        pooled_logit = config.voxel_prior_logit
        # Jeffreys smoothing, (c + 0.5) / (n + 1), not the raw rate. 58.6% of the
        # 4096 voxels are never positive anywhere in a real train split (measured
        # on fold P08's, 30371 episodes), and a raw rate sends every one of them
        # to the clamp floor. Smoothing puts them at logit -11.0 instead of -18.4:
        # still "essentially never", but on a scale the rest of the field shares.
        #
        # The alternative -- pulling unseen voxels to the pooled prior, i.e.
        # reading "never seen" as "no information" -- was tried and rejected. It
        # flattens the field to std 0.918 against the 1.866 of static spread the
        # QUERY pathway already supplies, so the arm would be weaker than the
        # thing it exists to overpower and would null out for a reason that has
        # nothing to do with the hypothesis. Smoothed it is std 2.938, i.e. 1.57x.
        counts = voxel_rate * n_train_episodes
        smoothed = (counts + 0.5) / (n_train_episodes + 1.0)
        field = np.log(smoothed / (1.0 - smoothed))
        bias_field = pooled_logit + args.voxel_bias_scale * (field - pooled_logit)
        print(f"[train] voxel bias field: {args.voxel_bias_init} "
              f"scale={args.voxel_bias_scale:+g} -- {int((voxel_rate <= 0).sum())}/{NUM_VOXELS} "
              f"voxels never positive in train (Jeffreys-smoothed, not clamped), "
              f"spread std={bias_field.std():.4f}, "
              f"range [{bias_field.min():+.3f}, {bias_field.max():+.3f}]")

    applied = model.apply_voxel_bias_init(bias_field)
    if applied and np.isscalar(bias_field):
        print(f"[train] voxel bias initialised to the prior logit "
              f"{config.voxel_prior_logit:+.4f} (p = {pooled_rate:.6f})")
    elif applied:
        r2 = getattr(model, 'voxel_bias_fit_r2', 1.0)
        print(f"[train] voxel bias initialised to a per-voxel field; the "
              f"{config.voxel_bias} parameterisation captured R^2={r2:.4f} of it")
        if args.voxel_bias_init == 'empirical' and config.voxel_bias == 'factorised':
            print("[train]   NOTE: --voxel_bias factorised has 48 parameters, so the "
                  "field was projected onto its additive part. Use --voxel_bias full "
                  "to hand the model the whole layout.")
    else:
        print("[train] voxel bias init: zero (legacy nn.Linear init); "
              "--voxel_bias_init prior applies the logit above")

    # Bias learning rate. Freezing is done here rather than in the optimizer so it
    # holds for any Trainer path; a positive value is picked up by CustomTrainer.
    if args.voxel_bias_lr == 0.0:
        frozen = model.set_voxel_bias_trainable(False)
        print(f"[train] voxel bias FROZEN at its init ({frozen} parameter tensor(s)); "
              "whatever it was given is what it keeps")
    elif args.voxel_bias_lr > 0:
        print(f"[train] voxel bias in its own optimizer group at lr={args.voxel_bias_lr}")
    if args.use_wandb:
        wandb.summary['train_positive_rate'] = pooled_rate
        wandb.summary['train_prior_logit'] = config.voxel_prior_logit
        for pid, (rate, count, n) in per_participant_rate.items():
            wandb.summary[f'train_positive_rate_{pid}'] = rate

    # --- Smoke test: one forward pass before training ---
    print("\n[SMOKE TEST] Running single forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    batch = torch_default_data_collator([train_dataset[0], train_dataset[1]])
    batch_on_device = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else
            {sk: sv.to(device) for sk, sv in v.items()} if isinstance(v, dict) else v)
        for k, v in batch.items()
    }
    with torch.no_grad():
        out = model(**batch_on_device)
    print(f"[SMOKE TEST] interaction logits {tuple(out['interaction'].shape)}, loss {out['loss'].item():.4f}")
    print("[SMOKE TEST] PASSED\n")
    model.train()

    training_args = TrainingArguments(
        output_dir=f"./results/{expt_name}/",
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        include_inputs_for_metrics=True,
        report_to="wandb" if args.use_wandb else "none",
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=torch_default_data_collator,
        compute_metrics=ComputeMetricsWrapper().compute_metrics,
    )
    trainer.set_voxel_bias_lr(args.voxel_bias_lr)   # before train(), which builds the optimizer
    # When there is no val split, eval_dataset already IS the test set, so we
    # don't register a separate test eval (it would duplicate it).
    if not no_val:
        trainer.set_test_dataset(test_dataset)
    trainer.evaluate()
    trainer.train()
