# KITScenes Training Plan

Plan and outcome for training AutoE2E on KITScenes for
[issue #168](https://github.com/autowarefoundation/auto_e2e/issues/168), on one
workstation. Every number is measured on this machine or read out of the
repository. Status: **two runs completed**; see §8.

For the step-by-step procedure, see [`Docs/running_experiments.md`](Docs/running_experiments.md).

---

## 1. What the model does

**Input** (one moment in time, at 10 Hz):

| Input | Shape | Plain meaning |
|---|---|---|
| Camera images | 6 × 3 × 256 × 256 | long-range front + 5 surround ring cameras |
| Navigation map | 14 × 256 × 256 | top-down map, one layer per feature (drivable area, lane boundary, crosswalk, …) |
| Route mask | 2 × 256 × 256 | which corridor to follow, and where the destination is |
| Ego history | 256 | the car's own speed/accel/yaw for the last 6.4 s |
| Visual history | 896 | compressed memory of recent frames (zeros when the World Model is off) |

**Output:** 64 timesteps × 2 numbers = the next **6.4 seconds**, as *acceleration*
and *curvature* (not x/y). Integrating those gives the path.

**Training signal:** the human driver's real future trajectory. SmoothL1 between
predicted and actual (accel, curvature). The KITScenes policy uses
`temporal_decay=0.99` with `mean_one` normalization and signal scales
(0.778 m/s², 0.035 1/m) — printed by every run.

**Score: ADE/FDE at 3 s.** That is what the official KITScenes benchmark uses —
`kitscenes_benchmark` hard-requires horizons `(3, 5)` and rejects anything else.
The training loop validates at 3 s and also records 1 s and 2 s.

---

## 2. The five steps of training

1. **Environment** — Python 3.12 + PyTorch built for the GPU.
2. **Pack the data** — raw per-frame JPEGs → WebDataset shards, one scene per shard.
3. **Audit** — mandatory navigation-quality gate; `train_il` refuses to run without it.
4. **Train** — read a batch → predict → compare to human → nudge weights. One pass
   = one *epoch*; a checkpoint per epoch.
5. **Evaluate** — ADE/FDE on scenes the model never trained on.

---

## 3. Hardware and what fits on it

Measured on this workstation (RTX 5080, 16 GB VRAM; 62 GB RAM; 1.7 TB free disk),
synthetic tensors on the 6-camera contract:

| batch size | peak VRAM | time per training step |
|---|---|---|
| 1 | 5.63 GB | 0.31 s |
| 2 | 12.08 GB | 0.36 s |
| 4 | **out of memory** | — |

**Decision: `batch_size=1` with `grad_accum_steps=4`.** Batch 2 fits but leaves
under 3 GB for the data loader. Batch 1 is also what `train_il` documents: a
batch-1 gradient is too noisy for the loss to descend past ~0.84, so gradients
accumulate over 4 micro-batches for one optimizer step — the batch-4 signal at
batch-1 memory.

Real throughput with `num_workers=4`: **10.36 samples/s** (residual),
**8.57 samples/s** (deformable).

**Do not enable fp16/AMP.** `train_il` ships `amp=False` deliberately: with fp16
the GradScaler detected inf/nan every step and skipped the optimizer update
*forever*, so the loss sat flat while the run looked healthy.

**One training process per GPU.** A second silently OOMs the first mid-run.

---

## 4. Which combinations work

| Axis | Value | Note |
|---|---|---|
| Backbone | `swin_v2_tiny` | repo default |
| Map fusion | `residual` / `deformable` | `cross_attn` is blocked, see below |
| Planner | `bezier` | `flow_matching` needs PR #172 |
| Dataset | KITScenes @ `6fde0034…` | pinned and enforced |
| BEV grid | 256 × 256 | from the navigation geometry contract; PR #188 proposes making it settable |

Verified by running each (`Tools/smoke_forward_pass.py`):

| backbone | fusion | planner | result |
|---|---|---|---|
| swin_v2_tiny | residual | bezier | works (73.6M params) |
| swin_v2_tiny | deformable | bezier | works (73.9M) |
| conv_next_v2_tiny | residual | bezier | works (73.9M) |
| res_net_50 | residual | bezier | works (70.1M) |
| swin_v2_tiny | **cross_attn** | bezier | **refuses to run** |
| swin_v2_tiny | residual | **flow_matching** | runs; wrong objective on `main` |

- **`cross_attn`** is dense O(N²) attention over BEV cells. At 256×256 that is
  65,536 cells, and the code hard-blocks above 4,096. `deformable` (4 sampled
  neighbours per cell) is the usable attention variant.
- **`flow_matching`** builds and produces gradients, but `train_il` on `main`
  SmoothL1-regresses an Euler-from-noise rollout rather than the velocity-MSE
  objective. **PR #172 wires `compute_planner_loss` and fixes this**; PR #189
  already trains FM through it. Treat the cell as blocked only until #172 lands.

**6 of 12 cells are trainable as specified today**, and 2 are done.

---

## 5. The data

| Fact | Number |
|---|---|
| Full train split | **2,619 GB** (533 archives) |
| Average archive | 4.91 GB (median 4.10, largest 21.6) |
| Scenes yielding zero samples | 129 of 533 |
| Total samples | 42,667 (38,847 train / 3,820 validation) |
| Average samples per usable scene | ~106 |

A scene needs 64 history + 1 + 64 future = **129 consecutive good frames**
(12.9 s at 10 Hz) for even one sample.

**Measured on the 60 scenes used here:** 272 GB raw → 381 MB packed (~170×);
5,666 samples; 56 partitions non-empty; 48 accepted by the navigation audit.

**Archive size predicts eligibility.** Selecting archives of 4–6 GB gave 93%
non-empty; a contributor sampling without that filter reported ~60% discarded.

Download is not the bottleneck — measured **111 MB/s**, so 60 archives took
12.3 min. Packing took 7.3 min for 14 scenes (~35 s/scene). The earlier estimate
of 1–4 h for packing was wrong by an order of magnitude.

---

## 6. The harness

Three registries already select components by name:

```
BACKBONE_REGISTRY     swin_v2_tiny | conv_next_v2_tiny | res_net_50
MAP_FUSION_REGISTRY   residual | cross_attn | deformable
PLANNER_REGISTRY      bezier | flow_matching
```

`AutoE2E.__init__` accepts `map_fusion_mode` and `planner_mode`; `train_il` never
passed them, so every sanctioned run silently trained `residual` + `bezier`.

**Implemented** in `Platform/pipelines/workflows.py` (43 lines): `MapFusion` and
`Planner` enums, both threaded through `train_il` and `wf_train_il`, recorded in
the checkpoint config so evaluation rebuilds the same architecture, and logged to
MLflow. Defaults reproduce the previous behaviour exactly.

**Overlap to resolve before opening a PR:** PR #172 (riita10069) already adds
`planner_mode: str` to `train_il`. Only `map_fusion_mode` is unique to this work;
the planner half should be dropped and coordinated with #172. PR #188 touches the
same signature block (`camera_bev_size`).

---

## 7. Why the posted results disagree

- different train/val splits (the maintainer asked for 585/65; the code enforces a
  frozen 533-scene split)
- different BEV resolutions (200×200, 450×300) — custom scripts bypassing the
  contract's 256×256
- fusion/planner settings that could not take effect through `train_il`
- fp16 on or off, which can silently disable learning
- warm-started runs compared against cold ones

**Every reported result should carry:** `group_digest`, `dataset_version`,
`packed_contract_digest`, backbone/fusion/planner, amp, effective batch size, and
ADE/FDE at 3 s. Same three digests ⇒ comparable.

---

## 8. Results

Two runs, 48 partitions, 6 held-out validation scenes,
`group_digest 6c5430fa…`, seed 149, batch 1×4, amp off, `validation_scope subset`.
Both early-stopped at epoch 8 (patience 5) with their best at epoch 3.

| | residual | deformable |
|---|---|---|
| Best ADE@3s | 1.179 m | **1.053 m** |
| Best FDE@3s | 3.481 m | **2.885 m** |
| Throughput | 10.36 /s | 8.57 /s |
| Wall clock | 59.3 min | 71.3 min |

![residual vs deformable](Docs/assets/issue168_ab_comparison.png)

**The difference is smaller than the noise inside either run** — deformable alone
swings from 1.05 to 2.07 m across epochs. With one seed and six validation scenes
this is not a demonstrated effect. Honest statement: deformable is not worse and
costs 17% throughput.

**Both runs overfit after epoch 3.** Training loss keeps falling (0.26 → 0.13)
while validation ADE does not improve — identically under both fusions. At 48
scenes the limiting factor is data quantity, not architecture.

### Neither run beats a constant-velocity baseline

Scored on the **identical 562 samples** — `sample_uid_digest 7682983a…` matches
the training run's exactly, so this is not an approximate comparison:

| | ADE@3s | FDE@3s |
|---|---|---|
| **constant velocity** (accel=0, curv=0) | **0.958 m** | **2.673 m** |
| deformable (best) | 1.053 m (+9.9%) | 2.885 m (+7.9%) |
| residual (best) | 1.179 m (+23.1%) | 3.481 m (+30.2%) |

`Model/evaluation/baselines.py` states the test: *"if the model barely beats
constant-velocity, perception isn't helping yet"*. Ours do not beat it at all.

This is consistent with PR #189 (0.883 m on a different split) and with every
ADE posted to the thread (1.16 m, 2.14 m, 3.62 m): **no reported run has yet
cleared the trivial baseline.** The open question is therefore not which fusion
is better but why the camera and map inputs are not contributing.

Reproduce with:

```bash
python Tools/experiments/run_baseline.py \
    --metadata /tmp/train/metadata.json --packed /path/to/packed
```

---

## 9. Environment gotchas

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: lark`, odd pytest plugin errors | ROS 2 on `PYTHONPATH` — use `env -u PYTHONPATH` |
| `pyproj==3.7.2` unsatisfiable | Python < 3.11 |
| `navigation rasterizer shared library is missing` | run `Model/navigation/native/build.py` |
| `validation manifest dataset version does not match packed shards` | packed with v2.2 instead of v3.3 |
| `requested val_fraction differs from the frozen manifest` | `val_fraction` is pinned at 0.1 |
| `MlflowException: … maintenance mode` | use `sqlite:///…`, not `file://` |
| `KeyError: 'MLFLOW_TRACKING_URI'` | read as bare `os.environ[...]` |
| STS / checkpoint upload failure | set `AUTO_E2E_CHECKPOINT_BUCKET` + `AWS_ENDPOINT_URL` |
| CUDA OOM mid-run | a second training process on the GPU |
| Loss perfectly flat, run looks fine | `amp=true` |

Also: the dataset is gated (accept terms **and** use a token with global gated-repo
read), and HuggingFace `main` has moved past the pinned revision `6fde0034…`.
Packing needs `kitscenes` (GitHub only) and `lanelet2` (wheel bundled in the SDK)
— without `lanelet2` map tiles silently become zero tensors.

---

## 10. Next

1. **Constant-velocity baseline on our split** — decides whether §8 is a result.
2. **Narrow the harness PR to `map_fusion_mode`**, rebase on `main`, coordinate
   with #172 and #188.
3. **Second seed** per fusion — without it the A/B is not publishable.
4. **More data** (150–200 scenes) — the overfitting in §8 says this is the real
   lever on absolute error.
5. **Report** with the full provenance tuple, labelled as a subset run: a subset
   holdout is not the frozen 40-scene validation set.
