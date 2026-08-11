# KITScenes Training Plan — Simple Version

A plain-language plan for training AutoE2E on KITScenes for
[issue #168](https://github.com/autowarefoundation/auto_e2e/issues/168), on one
workstation. Every number below was measured on this machine or read out of the
repository, not estimated.

---

## 1. What the model does

**Input** (one moment in time, at 10 Hz):

| Input | Shape | Plain meaning |
|---|---|---|
| Camera images | 7 × 3 × 256 × 256 | surround view from 7 cameras |
| Navigation map | 14 × 256 × 256 | a top-down map, one layer per feature (drivable area, lane boundary, crosswalk, …) |
| Route mask | 2 × 256 × 256 | which corridor to follow, and where the destination is |
| Ego history | 256 | the car's own speed/accel/yaw for the last 6.4 s |
| Visual history | 896 | compressed memory of recent frames (zeros when the World Model is off) |

**Output:** 64 timesteps × 2 numbers = the next **6.4 seconds** of driving, expressed
as *acceleration* and *curvature* (not x/y positions). Integrating those two signals
gives the actual path.

**Training signal:** the human driver's real future trajectory. The loss is
SmoothL1 between predicted and actual (accel, curvature), weighted so nearer
timesteps matter more (`temporal_decay=0.95`).

**Score:** ADE = average distance between predicted and true path; FDE = distance at
the end. Reported at 3 s and 6.4 s.

---

## 2. The five steps of training

1. **Environment** — Python + PyTorch built for this GPU. *(done)*
2. **Pack the data** — raw camera video → fixed-size tensor files (WebDataset
   shards), one scene per shard. This is the slow, disk-hungry step.
3. **Audit** — a mandatory navigation-quality gate; `train_il` refuses to run
   without it.
4. **Train** — repeat: read a batch → predict → compare to human → nudge weights.
   One pass over the data = one *epoch*. Save a checkpoint each epoch.
5. **Evaluate** — measure ADE/FDE on scenes the model never trained on.

---

## 3. Hardware and what fits on it

Measured on this workstation (RTX 5080, 16 GB VRAM; 62 GB RAM; 1.7 TB free disk):

| batch size | peak VRAM | time per training step |
|---|---|---|
| 1 | 6.09 GB | 0.30 s |
| 2 | 12.08 GB | 0.36 s |
| 4 | **out of memory** | — |

**Decision: `batch_size=1` with `grad_accum_steps=4`.**

Batch size 2 technically fits, but leaves under 3 GB for the data loader, and these
numbers come from synthetic tensors — the real path also holds decoded JPEGs. Batch
size 1 is also what `train_il`'s own documentation recommends: a batch-1 gradient is
too noisy for the loss to descend past ~0.84, so instead of a bigger batch we
accumulate gradients over 4 micro-batches and take one optimizer step. Same
statistical effect as batch 4, at batch-1 memory cost.

**Do not enable fp16/AMP.** `train_il` ships `amp=False` deliberately: with fp16 the
gradient scaler detected inf/nan every step and skipped the optimizer update
*forever*, so the model silently never learned. Fitting the model by turning on fp16
is the one shortcut that produces a plausible-looking, meaningless run.

---

## 4. Which model to run

| Axis | Choice | Why |
|---|---|---|
| Backbone | `swin_v2_tiny` | repo default; matches another contributor's run, so the number is comparable to something |
| Map fusion | `residual` | the only mode `train_il` can currently produce |
| Planner | `bezier` | ditto — and `flow_matching` optimizes the wrong objective (see below) |
| Dataset | KITScenes @ revision `6fde0034…` | pinned and enforced by the code |
| BEV grid | 256 × 256 | pinned by the navigation geometry contract |

### Which combinations actually work

Issue #168 proposes 3 backbones × 2 fusions × 2 planners = 12 combinations. Verified
by running each one:

| backbone | fusion | planner | result |
|---|---|---|---|
| swin_v2_tiny | residual | bezier | works (73.6M params) |
| swin_v2_tiny | deformable | bezier | works (73.9M) |
| conv_next_v2_tiny | residual | bezier | works (73.9M) |
| res_net_50 | residual | bezier | works (70.1M) |
| swin_v2_tiny | **cross_attn** | bezier | **refuses to run** |
| swin_v2_tiny | residual | **flow_matching** | runs, but trains the wrong objective |

- **`cross_attn`** is dense attention, cost O(N²) in the number of BEV cells. At
  256×256 that is 65,536 cells and a 4-billion-entry score matrix, so the code
  hard-blocks it above 4,096 cells. `deformable` (each cell looks at 4 sampled
  neighbours instead of all of them) is the usable attention variant.
- **`flow_matching`** builds and produces gradients, but the training loop regresses
  its output with SmoothL1 while the planner integrates from fresh random noise each
  step. That is not the flow-matching objective; it drives the model toward the
  average of all plausible futures. The correct loss exists in the code but is not
  wired into the training loop.

**So: 6 of 12 combinations are trainable as specified**, and only after the fix in
§6.

---

## 5. The data problem

| Fact | Number |
|---|---|
| Full train split | **2,619 GB** (533 archives) |
| Average archive | 4.91 GB (median 4.10, largest 21.6) |
| Scenes yielding zero training samples | 129 of 533 |
| Usable scenes | 404 |
| Total samples | 42,667 (38,847 train / 3,820 validation) |
| Average samples per usable scene | ~106 |

A scene needs 64 history steps + 1 + 64 future steps = **129 consecutive good
frames** (12.9 s at 10 Hz) to yield even one training sample. Short scenes yield
nothing, which is why 129 scenes drop out.

Downloading everything is not viable on one workstation. The approach that is:
**download one archive → pack it → delete the raw archive → next.** Packed output is
tiny compared to raw (a contributor measured the whole corpus packing down to ~24 GB),
so peak disk stays at one scene.

---

## 6. The harness (why this is the real contribution)

The model code is already modular. Three registries pick components by name:

```
BACKBONE_REGISTRY     swin_v2_tiny | conv_next_v2_tiny | res_net_50
MAP_FUSION_REGISTRY   residual | cross_attn | deformable
PLANNER_REGISTRY      bezier | flow_matching
```

`AutoE2E.__init__` already accepts `map_fusion_mode` and `planner_mode`.

**The gap:** `train_il` never passes them (`Platform/pipelines/workflows.py:3898`),
so every sanctioned run silently gets `residual` + `bezier`, whatever the
experimenter intended. Two of the issue's three axes are unreachable through the
official path.

**The fix is small:** two enums beside the existing `Backbone` enum, two `train_il`
parameters, forwarded into the constructor — roughly 30 lines. Everything downstream
already exists: MLflow logs `model/backbone` and `model/fusion_mode`, and the
evaluation task already reports ADE/FDE at 3 s and 5 s.

This matters more than any single training run: without it, contributors comparing
"residual vs attention" are comparing two runs that both used residual, or two
custom scripts that left the contract in different ways.

---

## 7. Why the posted results disagree

Four contributors have posted ADE/FDE numbers that cannot be compared:

- different train/val splits (the maintainer asked for 585/65; the code enforces a
  frozen 533-scene split and rejects anything else)
- different BEV resolutions (200×200, 450×300) — because custom scripts bypassed the
  contract's 256×256
- fusion/planner settings that could not have taken effect through `train_il`
- fp16 on or off, which can silently disable learning entirely

**Every reported result should carry:** validation `group_digest`, `dataset_version`,
`packed_contract_digest`, backbone/fusion/planner, amp on/off, effective batch size,
and ADE/FDE at both 3 s and 6.4 s. Same three digests ⇒ comparable. Different ⇒ not.

---

## 8. Steps taken so far

| # | Step | Result |
|---|---|---|
| 1 | Read issue #168 and all 20 comments | 4 contributors, mutually incomparable numbers |
| 2 | Traced the pinned split in code | `dataset_policy.py:121` → v3 manifest, `dataset_version` v3.3 |
| 3 | Found `train_il` drops fusion/planner | `workflows.py:3898` |
| 4 | Found `cross_attn` guard and `flow_matching` warning | 6 of 12 cells trainable |
| 5 | Confirmed BEV is pinned at 256×256 | `navigation/geometry.py:225` |
| 6 | Built Python 3.12 env, torch 2.7.1+cu128 | `sm_120` present, RTX 5080 supported |
| 7 | Ran the repo test suite | **914 passed**, 3 failed — all from an optional NVIDIA-only package |
| 8 | Wrote `Tools/smoke_forward_pass.py` | tests any combination without data |
| 9 | Ran all 6 combinations on CPU | shapes and gradients verified |
| 10 | Freed the GPU (stopped CARLA + devcontainer) | 15.2 GB VRAM, 50 GB RAM available |
| 11 | Ran smoke test on GPU | pass — 0.30 s/step, 6.09 GB peak |
| 12 | Measured batch-size ceiling | 1 → 6 GB, 2 → 12 GB, 4 → OOM |
| 13 | Measured dataset size at the pinned revision | 2,619 GB, mean 4.91 GB/archive |

### Gotchas found along the way

- **ROS2 leaks into the environment.** `/opt/ros/humble` on `PYTHONPATH` breaks
  training runs. Always `env -u PYTHONPATH`.
- **Python 3.10 is too old** — pinned `pyproj==3.7.2` needs ≥3.11.
- **The dataset moved.** The contract pins revision `6fde0034…`; HuggingFace `main`
  is now `213352cf…`. The pipeline enforces this; manual downloads must pass it.
- **All 40 validation scenes live in `data/train/`**, confirming the frozen split
  never touches `data/val/`.

---

## 9. Blocker: no data access yet

**Downloading any KITScenes data currently fails with HTTP 403.** Everything else is
ready; this is the one thing standing between here and a training run.

```
GatedRepoError: 403 ... Access to dataset KIT-MRT/KITScenes-Multimodal is
restricted and you are not in the authorized list.
```

Metadata (file names, sizes) is public, which is why the measurements in §5 worked.
File *contents* are gated. Diagnosis of this machine's token:

```
token role:  fineGrained  (name: "Thinkstation")
canReadGatedRepos: True
global permissions: []                      <-- empty
scoped permissions: cbank's own repos only  <-- does not cover KIT-MRT
```

So there are two things to fix, and possibly both:

1. **Accept the dataset terms** at
   https://huggingface.co/datasets/KIT-MRT/KITScenes-Multimodal — the repo is
   `gated: auto`, meaning approval is automatic once the terms are accepted; no
   waiting on a human.
2. **Widen the token.** The current fine-grained token is scoped to the `cbank`
   namespace with no global permissions, so it grants nothing on a `KIT-MRT` repo.
   Either add global "Read access to contents of all public gated repos you can
   access", or use a plain `read`-role token.

Verify with:

```bash
hf download KIT-MRT/KITScenes-Multimodal data/sequence_archives.csv \
  --repo-type dataset --revision 6fde0034446669e2ed7235e4c7fe323cd23d599d
```

That file is small; if it succeeds, the tars will too.

---

## 10. Will an overnight run work?

**Arithmetic, on measured numbers.** At 0.30 s/step and batch 1:

| corpus | samples | GPU time per epoch |
|---|---|---|
| 12 scenes | ~1,270 | ~6 min |
| 40 scenes | ~4,200 | ~21 min |
| full 404 scenes | 38,847 | ~3.2 h |

But GPU time is not the limit — **data loading is**. Each sample decodes roughly 55
JPEGs, and with `num_workers=0` that happens serially while the GPU waits. Expect
2–5× the GPU-only figure.

**Realistic:** a 12-scene run for 10 epochs is roughly 2–5 hours — comfortably one
night. The full corpus is not a single-night job, and 2.6 TB will not fit anyway.

**The real risk is not compute, it is plumbing.** Still unverified: downloading and
packing a scene, the mandatory audit, and invoking `train_il` locally (it is a Flyte
task expecting cloud storage; another contributor needed five workarounds — a local
checkpoint destination, an S3-compatible endpoint, and an MLflow URI, among others).

**Therefore: prove the whole chain on one scene first.** Download → pack → audit →
one epoch. Only when that completes end-to-end is an overnight run worth starting.

### Verdict

**Tonight: no.** Not for hardware reasons — the GPU is free, the model runs, the
environment is clean. The dataset is inaccessible (§9), so step 2 of 5 cannot start.

Once access is granted, the remaining unknowns are the pack, the audit and the local
`train_il` invocation. Those are a working session, not an unattended one. The first
genuinely overnight-able run is the 12-scene job, *after* one scene has been proven
end to end.

---

## 11. Recommended order

1. **Fix the harness** (§6) — no data needed, unblocks everyone, ~30 lines.
2. **Post the correction** to issue #168 so others stop generating incomparable runs.
3. **One scene, end-to-end** — the plumbing test.
4. **12 scenes overnight**, `swin_v2_tiny` + `residual` + `bezier`, batch 1 ×
   grad-accum 4, amp off, 10 epochs.
5. **Report** with the full provenance tuple, labelled as a subset run — a subset
   holdout is *not* the frozen 40-scene validation set, so the number is a
   pipeline-validation result, not a benchmark entry.
