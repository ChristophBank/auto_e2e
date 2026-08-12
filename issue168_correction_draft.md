# Draft: comment for issue #168

Rewritten to lead with the constant-velocity result, which changes what the
other findings mean. Everything else was either superseded by a PR (#172, #188)
or folded into PR #192's description.

---

## No reported run beats a constant-velocity baseline

I trained two configurations overnight and then scored
`evaluation.baselines.constant_velocity_baseline` — `accel = 0, curv = 0`, no
perception at all — on the **identical validation samples**, through the same
`_evaluate_open_loop` path:

| | ADE@3s | FDE@3s |
|---|---|---|
| **constant velocity** | **0.958 m** | **2.673 m** |
| swin_v2_tiny + deformable + bezier | 1.053 m | 2.885 m |
| swin_v2_tiny + residual + bezier | 1.179 m | 3.481 m |

Not an approximate comparison: the baseline's `sample_uid_digest`
(`7682983a…`) equals the training run's exactly — same 562 samples, same metric
contract, same aggregation.

`Model/evaluation/baselines.py` states the test it exists for:

> *if the model barely beats constant-velocity, perception isn't helping yet*

Ours do not beat it at all. And this does not look specific to my setup:

| Source | ADE@3s |
|---|---|
| constant velocity, my split | **0.958 m** |
| constant velocity, #189's split | **0.883 m** |
| my deformable run | 1.053 m |
| exp-2 (deformable, 50 scenes) | 1.16 m |
| earlier thread results | 2.14 m, 3.62 m |

Two independently measured baselines on different splits agree to within 0.08 m,
and **every learned number posted to this issue so far sits above both**. @FLagbusted
reached the same conclusion at epoch 0 in #189; this shows it also holds after
training converges and early-stops.

I do not think this means the architecture is wrong. It means we currently have
no evidence that the camera and map inputs contribute anything, and the
residual-vs-attention question is being asked one level too early.

### Reproducing it on your own runs

`Tools/experiments/run_baseline.py` (happy to open a PR for it) reads the
validation group UIDs out of a finished run's `metadata.json`, rebuilds the same
loader, and scores a stub module in the model's place. It compares the resulting
`sample_uid_digest` with the run's and refuses to present the comparison as valid
if they differ — the check costs nothing and makes the number arguable rather
than assumed.

```bash
python Tools/experiments/run_baseline.py \
    --metadata /path/to/metadata.json --packed /path/to/packed
```

`hold_last_action_baseline` is also in `baselines.py` and is the stronger ego-only
reference; worth reporting alongside.

### What I would suggest next

A **camera ablation** — the same training run with the camera tensors zeroed. If
ADE does not get worse, the model demonstrably ignores the images, and we would
know that rather than suspecting it. It costs one run.

Two observations that may be related, both from my runs and consistent with the
WG note that *"network quality depends on input data quality, not just
architecture changes"*:

- **Both runs overfit after epoch 3** — training loss falls 0.26 → 0.13 while
  validation ADE flatlines, identically under both fusions. At 48 scenes the
  binding constraint looks like data quantity.
- The **A/B difference is smaller than the noise inside one run**: deformable
  swings 1.05 → 2.07 m ADE across its own epochs while the gap to residual is
  0.13 m. One seed and six validation scenes cannot separate them.

---

## Two smaller things

**The pinned split manifest is v3, not v2.** `dataset_policy.py:121` resolves
`kitscenes_train_dev_v3.json` (`dataset_version` v3.3). All three manifests share
`split_id` and `eligible_group_uid_digest` but differ in `dataset_version` and
`packed_contract_digest` — so matching the validation `group_digest` alone is not
sufficient to prove two runs are comparable. The reportable tuple is
**`group_digest` + `dataset_version` + `packed_contract_digest`**, all three
printed by the run itself.

**Archive size predicts scene eligibility.** A scene needs 129 consecutive good
frames to yield a single sample. Selecting archives of 4–6 GB gave 56 of 60
non-empty (93%); sampling without that filter reportedly discards ~60%. Cheap way
to avoid downloading scenes that pack to nothing.

---

## Suggested reporting convention

So the next batch of numbers is comparable:

```
scope:                  full | subset (+ packed partition count)
group_digest:           <from the "Validation split: …" log line>
dataset_version:        <from the packed manifest>
packed_contract_digest: <from the packed manifest>
backbone / map_fusion_mode / planner_mode
amp:                    on | off
effective batch:        batch_size x grad_accum_steps
ADE/FDE @ 3s            (+ the constant-velocity baseline on the same samples)
```

and that anything not produced by `train_il` is labelled as such.

Full run details, measured stage costs and the environment traps I hit are in
[TRAINING_PLAN.md / SESSION_FINDINGS.md — link once pushed].

PR #192 makes `map_fusion_mode` selectable through `train_il`; without it the
Residual-vs-Attention axis of this issue cannot be run on the sanctioned path at
all.
