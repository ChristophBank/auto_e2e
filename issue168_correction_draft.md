# Ready to post — issue #168

Copy everything below the line. Attach `Docs/assets/issue168_ab_comparison.png`.

---

## Two overnight runs on the `train_il` contract — and neither beats constant velocity

I trained two configurations of the Reactive branch overnight on a single
workstation, differing in exactly one parameter, then scored a training-free
baseline on the same samples. The baseline result is the part worth discussing.

### Configuration

Both runs are identical except for `map_fusion_mode`.

**Data**

| | |
|---|---|
| Dataset | `KIT-MRT/KITScenes-Multimodal` @ `6fde0034446669e2ed7235e4c7fe323cd23d599d` |
| Packed partitions | 60 (one scene per partition) |
| Non-empty | 56 — 4 scenes yielded no sample |
| Accepted by the navigation audit | 48 for the optimizer |
| Training samples / epoch | 4,293 |
| Validation | 6 scene groups, 562 samples |
| `validation_scope` | `subset` (a partial corpus cannot satisfy the frozen 533-partition check) |
| `val_fraction` | 0.1 (pinned by the frozen manifest) |
| `group_digest` | `6c5430fafda558b6f61478c8d124a2cf9e5bc2b9f121b3815a1c212b1c8c045f` |
| `dataset_version` | `v3.3` |
| `packed_contract_digest` | `6fb9d857d877e570522a59a38097b0f6b3320d8f702bbd05ec9670faa4be6d96` |

Scenes were selected by archive size (4–6 GB). A scene needs 129 consecutive good
frames for a single sample; that filter gave 56/60 non-empty, where sampling
without it reportedly discards ~60%.

**Model**

| | |
|---|---|
| Backbone | `swin_v2_tiny`, ImageNet-pretrained |
| Views | 6 cameras @ 256×256 (long-range front + 5 ring) |
| Navigation | 14-channel map context + 2-channel route mask, `route_valid=True` |
| BEV grid | 256×256, geometry `kitscenes-v3-bev-1m-v1` |
| Map-BEV fusion | **`residual`** vs **`deformable`** ← the only difference |
| Planner | `bezier` |
| World model / reasoning / route consistency | off |
| Parameters | 73.6M / 73.9M |

**Training**

| | |
|---|---|
| Objective | `trajectory_imitation_v1`, SmoothL1 |
| Loss policy | `temporal_decay=0.99`, `mean_one`, signal scales (0.778 m/s², 0.035 1/m) |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-2, grad clip 1.0 |
| Scheduler | `ReduceLROnPlateau`, factor 0.5, patience 1 |
| Batch | `batch_size=1` × `grad_accum_steps=4` (effective 4) |
| AMP | **off** |
| Seed | 149, both runs |
| Epochs | 20 configured, `early_stopping_patience=5` |
| Hardware | RTX 5080, 16 GB — peak 5.6 GB at batch 1 |

Both runs stopped at **epoch 8** with their best checkpoint at **epoch 3**.

### Results

| | residual | deformable |
|---|---|---|
| Best ADE@3s | 1.179 m | 1.053 m |
| Best FDE@3s | 3.481 m | 2.885 m |
| Throughput | 10.36 samples/s | 8.57 samples/s |
| Wall clock, 8 epochs | 59.3 min | 71.3 min |

I would not read the fusion difference as an effect. Deformable swings from 1.05
to 2.07 m ADE across its own epochs, so the spread *inside* one run is larger
than the 0.13 m gap *between* the runs. With one seed and six validation scenes,
the defensible statement is: deformable is not worse and costs 17% throughput.

### Neither run beats a constant-velocity baseline

I then scored `evaluation.baselines.constant_velocity_baseline` — `accel = 0`,
`curv = 0`, no perception — through the same `_evaluate_open_loop` path:

| | ADE@3s | FDE@3s |
|---|---|---|
| **constant velocity** | **0.958 m** | **2.673 m** |
| deformable (best) | 1.053 m | 2.885 m |
| residual (best) | 1.179 m | 3.481 m |

This is not an approximate comparison: the baseline's `sample_uid_digest`
(`7682983a…`) equals the training run's exactly — same 562 samples, same metric
contract, same aggregation.

`Model/evaluation/baselines.py` states the test it exists for:

> *if the model barely beats constant-velocity, perception isn't helping yet*

Ours do not beat it at all. And it does not look specific to my setup:

| Source | ADE@3s |
|---|---|
| constant velocity, my split | **0.958 m** |
| constant velocity, #189's split | **0.883 m** |
| my deformable run | 1.053 m |
| exp-2 (deformable, 50 scenes) | 1.16 m |
| earlier results in this thread | 2.14 m, 3.62 m |

Two independently measured baselines on different splits agree to within 0.08 m,
and every learned number posted here so far sits above both. @FLagbusted reached
the same conclusion at epoch 0 in #189 — this shows it also holds after training
converges and early-stops.

I do not think this means the architecture is wrong. It means we have no evidence
yet that the camera and map inputs contribute anything, and that the
residual-vs-attention question is being asked one level too early.

### Related observation

Both runs overfit after epoch 3 — training loss falls 0.26 → 0.13 while
validation ADE flatlines, identically under both fusions. At 48 scenes the
binding constraint looks like data quantity, which matches the WG note that
*"network quality depends on input data quality, not just architecture changes"*.

### Suggestions

1. **A camera ablation** — the same run with the camera tensors zeroed. If ADE
   does not get worse, the model demonstrably ignores the images and we would
   know it rather than suspect it. One run.
2. **Report the baseline next to every result.** It costs a few minutes and turns
   an ADE into a statement about whether perception is doing anything.
   `hold_last_action_baseline` is also in `baselines.py` and is the stronger
   ego-only reference.
3. **Quote all three digests**, not just the validation `group_digest` —
   `group_digest` + `dataset_version` + `packed_contract_digest`. The manifests
   share a `split_id` and an `eligible_group_uid_digest` while differing in the
   other two, so the group digest alone does not establish comparability.

I have a script that reads the validation group UIDs out of a finished run's
`metadata.json`, rebuilds the same loader, scores a stub module in the model's
place, and refuses to present the comparison as valid if the `sample_uid_digest`
does not match. Happy to open a PR for it if useful.

(Separately, PR #192 makes `map_fusion_mode` selectable through `train_il` — the
two runs above required it, since `train_il` previously always built `residual`.)
