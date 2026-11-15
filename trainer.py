from transformers import Trainer
from typing import Optional
import pickle
import os

# Parameters that get their own, much larger learning rate. See
# create_optimizer() below for why this exists at all.
# NOTE these are CANDIDATES: most runs build only one or two of them, since each is
# created under its own config flag. Assert against the ones a build actually has
# (see smoke_token_scale.py), never against len(FAST_PARAMS).
# spatial_W_r is the one non-scalar entry (a [H/2, 3] frequency matrix, not an
# amplitude): the same travel-budget argument applies to it -- at the production
# lr=1e-5 it moved from std=10.0 to std=9.6-10.1 over a full 1899-step run,
# i.e. not at all, so its init IS the frequency content unless it rides here too.
FAST_PARAMS = {'time_gain', 'spatial_gain', 'class_gain', 'spatial_W_r'}
FAST_LR = 1e-3


class CustomTrainer(Trainer):
    """HuggingFace Trainer subclass that adds three behaviours:

    1. Test-set evaluation alongside validation: every time the base Trainer runs an
       eval loop (each eval_steps interval, plus the optional eval_on_start tick), we
       additionally run a full evaluation loop on the held-out test split and log its
       metrics. (The base Trainer only supports a single eval_dataset; this wires in
       a second one.)

    2. Checkpoint augmentation: after saving a model checkpoint, also writes the most recent
       eval predictions, labels, and sample indices to pickle files inside the checkpoint directory.
       This lets you reconstruct per-sample results offline without re-running inference.

    3. A separate, faster learning rate for the scalar gains (see create_optimizer).
    """

    def create_optimizer(self):
        """Put the scalar gains (`time_gain`, `spatial_gain`) in their own group at FAST_LR.

        Why this is not premature tuning. AdamW's update is ~lr per step whatever
        the gradient's size, so at the production LR=1e-5 with linear decay a
        SCALAR can travel at most ~0.1 over a full 19.4k-step run -- from an init
        of 8.0 it could reach 7.9 or 8.1 and nothing else. `time_gain` would be
        learnable in name only, and its init would silently be the experiment.

        That matters here because the quantity it controls is the one under test.
        The 2026-08-12 te10 runs pinned the time code at norm 0.707 (2.6% of a
        location token) and measured nothing -- mean best PR-AUC .1288 against
        .1301 with no encoding at all. Guessing a replacement number just moves
        where the null could hide.

        The travel budget is the sum of the per-step lr under linear decay,
        lr*T/2, so at 1e-3 over ~19.4k steps it is ~9.7 -- from an init of 8.0 the
        scalar can reach anywhere in [0, ~17.7], i.e. from silent to 64% of a
        LayerNorm'd token's norm (27.71). That makes where it settles a
        measurement rather than an assumption: drifting to ~0 says this task does
        not want recency, and a value it holds says how much it does. If it pins
        at the ceiling instead, the answer is "more than this run could express"
        and the next one needs a larger init or a faster group.

        `spatial_gain` (--position_encoding fourier) is in for the same reason and
        needs it more: it inits at 48.0, so the ~0.1 of travel available at 1e-5
        is 0.2% -- learnable in name only. At 1e-3 it can reach roughly [38, 58],
        which spans "quieter than the index tables it replaced" to "louder", making
        the code's magnitude a measurement rather than the 48.0 we assumed.

        `spatial_W_r` (the [H/2, 3] frequency matrix `spatial_gain` scales) rides
        here too, for the same reason under a stricter budget: measured on a
        finished --position_encoding fourier run, its per-axis std moved from a
        10.0 init to 9.6-10.1 over 1899 steps at 1e-5 -- not at all. Unlike a
        scalar, decaying it toward 0 would collapse every frequency to DC, so it
        is exempted from weight decay the same way the gains are.

        Everything else keeps args.learning_rate exactly, so a run with
        time_encoding_gain='fixed' and position_encoding='none' (both defaults) has
        no fast group at all and is optimiser-identical to every run before this
        existed.
        """
        if self.optimizer is not None:
            return self.optimizer

        # --voxel_bias_lr > 0 puts the bias in its own group, for exactly the
        # reason the gains have one: measured over 27 finished voxel_query runs
        # it drifts at most 0.036 from its init at lr=1e-5, so every result so
        # far measured the INIT, not a learned prior. Set by train.py; absent on
        # every run before this existed, which then takes the branch below
        # unchanged. Freezing (lr == 0) is handled in the model via
        # requires_grad, so those tensors never reach this loop.
        bias_lr = getattr(self, 'voxel_bias_lr', -1.0)
        bias_names = set()
        if bias_lr > 0:
            bias_names = {n for n, _ in self.model.voxel_bias_parameters()}

        fast, normal, fast_names = [], [], []
        bias, bias_found = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in FAST_PARAMS:
                fast.append(param)
                fast_names.append(name)
            elif name in bias_names:
                bias.append(param)
                bias_found.append(name)
            else:
                normal.append(param)
        if not fast and not bias:
            return super().create_optimizer()

        # Match the base class's decay/no-decay split for the normal group rather
        # than reimplementing it: HF excludes bias and LayerNorm weights from
        # weight decay, and silently dropping that would change every other run.
        decay = self.get_decay_parameter_names(self.model)
        named = dict(self.model.named_parameters())
        cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        held = FAST_PARAMS | bias_names
        groups = [
            {'params': [p for n, p in named.items()
                        if n in decay and n not in held and p.requires_grad],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in named.items()
                        if n not in decay and n not in held and p.requires_grad],
             'weight_decay': 0.0},
        ]
        if fast:
            # A scalar amplitude is not a weight: decaying it would pull the time
            # code back toward silence, which is the failure mode being fixed.
            groups.append({'params': fast, 'weight_decay': 0.0, 'lr': FAST_LR})
        if bias:
            # Never decayed. HF already excludes anything named *bias* from decay,
            # and pulling this one toward 0 would drag the output back to
            # sigmoid(0)=0.5 against a 1:200 positive rate.
            groups.append({'params': bias, 'weight_decay': 0.0, 'lr': bias_lr})
        self.optimizer = cls(groups, **kwargs)
        if bias_found:
            print(f"[optimizer] {len(bias_found)} voxel-bias parameter(s) at lr={bias_lr} "
                  f"({', '.join(sorted(bias_found))})")
        # The names that actually MATCHED, not FAST_PARAMS itself: only one of the
        # two gains exists in most runs, and printing the candidate set made the
        # log say "1 parameter(s) (spatial_gain, time_gain)".
        if fast_names:
            print(f"[optimizer] {len(fast)} parameter(s) at lr={FAST_LR} "
                  f"({', '.join(sorted(fast_names))})")
        print(f"[optimizer] everything else at lr={self.args.learning_rate}")
        return self.optimizer

    def set_voxel_bias_lr(self, lr):
        """Register a private learning rate for the voxel bias (see create_optimizer).

        An attribute rather than a TrainingArguments field because HF rejects
        unknown kwargs there, the same reason set_test_dataset exists.
        """
        self.voxel_bias_lr = float(lr)

    def set_test_dataset(self, test_dataset):
        """Register the test dataset to be evaluated alongside validation.

        Args:
            test_dataset: torch Dataset or Subset — the held-out test split
        """
        self.test_dataset = test_dataset
        self._last_test_step = -1  # Tracks the last global_step at which test eval was run (avoids double-eval)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override the base evaluate() to also run test evaluation alongside val.

        The base class calls this method at each eval_steps interval (and at step 0
        when eval_on_start=True). We intercept it to:
          1. Run the standard validation evaluation (delegated to super()).
          2. Run test evaluation at the same cadence and merge test metrics in.

        Args:
            eval_dataset:      Dataset or None — if None, uses self.eval_dataset (val set)
            ignore_keys:       list of str or None — output keys to exclude from metric collection
            metric_key_prefix: str — prefix applied to all returned metric names ("eval" or "test")

        Returns:
            metrics: dict of str → float — all eval (and optionally test) metrics
        """
        # Run standard val evaluation; metrics keys are prefixed with "eval_"
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Run test evaluation alongside val. Guarded by _last_test_step so that if
        # evaluate() is somehow called twice at the same global_step we don't double up.
        if (hasattr(self, 'test_dataset') and
                metric_key_prefix == "eval" and
                self.state.global_step != self._last_test_step):
            self._last_test_step = self.state.global_step
            # Re-use the base evaluate() with the test dataset and a "test_" prefix
            test_metrics = super().evaluate(self.test_dataset, ignore_keys, metric_key_prefix="test")
            metrics.update(test_metrics)  # Merge test metrics into the return dict
            self.log(test_metrics)        # Also push test metrics to WandB / logger

        return metrics

    def evaluation_loop(self, *args, **kwargs):
        """Override the evaluation loop to capture raw predictions, labels, and indices.

        The base evaluation_loop runs inference on the entire eval DataLoader, calls
        compute_metrics, and returns an EvalLoopOutput named tuple.  We intercept this to:
          - Store predictions (as numpy arrays) on self for use in _save()
          - Strip the non-JSON-serialisable 'indexes' array from the metrics dict before the
            base class tries to serialise metrics to trainer_state.json

        Returns:
            output: EvalLoopOutput — same as base class, but with *_indexes removed from metrics
        """
        output = super().evaluation_loop(*args, **kwargs)

        # output.predictions: np.ndarray or tuple of np.ndarrays — model outputs collected over eval set
        # output.label_ids:   dict or np.ndarray — ground-truth labels collected over eval set
        # output.metrics:     dict of str → scalar float — results from compute_metrics()
        self.custom_current_predictions = output.predictions
        self.custom_current_labels = output.label_ids

        # compute_metrics returns 'indexes': np.ndarray [N] — the sample indices used in evaluation.
        # This is useful for offline analysis but is not JSON-serialisable, so it must be removed
        # from output.metrics before Trainer tries to write trainer_state.json
        indexes_key = next((k for k in output.metrics if k.endswith('_indexes')), None)
        # indexes_key: str (e.g. "eval_indexes" or "test_indexes") or None if key not present
        self.custom_current_indexes = output.metrics.pop(indexes_key, None)
        # custom_current_indexes: np.ndarray [N] int — sample indices, or None

        return output

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override checkpoint saving to also persist predictions, labels, and indices.

        The base _save() writes model weights, config, and tokenizer to output_dir.
        We extend it to additionally write pickle files for the most recently evaluated
        predictions, ground-truth labels, and sample indices.  These files are placed
        inside the same checkpoint directory so they are versioned alongside the weights.

        Files written (when data is available):
            predictions.pkl — np.ndarray or tuple of np.ndarrays
            labels.pkl      — dict with 'interaction': np.ndarray [N_eval, 4096] int
            indexes.pkl     — np.ndarray [N_eval] int — sample indices
        """
        super()._save(output_dir, state_dict)   # Write model weights and config files

        if hasattr(self, 'custom_current_predictions') and self.custom_current_predictions is not None:
            with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump(self.custom_current_predictions, f)

        if hasattr(self, 'custom_current_labels') and self.custom_current_labels is not None:
            with open(os.path.join(output_dir, 'labels.pkl'), 'wb') as f:
                pickle.dump(self.custom_current_labels, f)

        if hasattr(self, 'custom_current_indexes') and self.custom_current_indexes is not None:
            with open(os.path.join(output_dir, 'indexes.pkl'), 'wb') as f:
                pickle.dump(self.custom_current_indexes, f)
