# Session findings — KITScenes training for issue #168

Compact record of what was established on 2026-08-11/12, with evidence pointers.
Everything here was verified by running it or reading the code, not inferred.

---

## A. Code findings

| # | Finding | Evidence |
|---|---|---|
| A1 | `train_il` never forwarded `map_fusion_mode` or `planner_mode`, so **every** run silently trained `residual` + `bezier` regardless of intent. Two of issue #168's three axes were unreachable. | `workflows.py:3898`; defaults at `auto_e2e.py:28,30` |
| A2 | `cross_attn` map fusion **refuses to run** above 4,096 BEV tokens. The contract grid is 256×256 = 65,536. `deformable` is the only usable attention variant. | `map_bev_fusion/cross_attention_fusion.py:70` |
| A3 | `flow_matching` runs and produces gradients but optimizes the **wrong objective** on `main` — SmoothL1 on an Euler-from-noise rollout, not velocity-MSE. **PR #172 fixes this.** | `trajectory_planning/__init__.py` RuntimeWarning; PR #172 |
| A4 | The pinned split is **v3**, not the v2 cited in the issue thread. All three manifests share `split_id` and `eligible_group_uid_digest` but differ in `dataset_version` (v2.2/v3.1/v3.3) and `packed_contract_digest`. | `dataset_policy.py:121` |
| A5 | Comparability therefore needs **three** values, not one digest: `group_digest` + `dataset_version` + `packed_contract_digest`. | derived from A4 |
| A6 | BEV is **pinned at 256×256** for KITScenes by the navigation geometry contract — not the 450×300 `BEVViewFusion` default, which only reaches L2D. PR #188 proposes making it settable. | `workflows.py:3387`; `navigation/geometry.py:225` |
| A7 | Training validation evaluates **30 of 64 predicted steps** (3.0 s). The official benchmark hard-requires horizons `(3, 5)` and raises otherwise — so 3 s is the metric by design, not a defect. | `workflows.py:914`; `evaluation/kitscenes_benchmark.py:198` |
| A8 | `val_fraction` is checked against the frozen manifest and **must be 0.1**. More validation scenes require more partitions, not a larger fraction. | `dataset_policy.py`; run error |
| A9 | KITScenes ships **9 cameras**; the contract uses **6**: the 18.2 Mpx long-range front replaces `camera_ring_front`, and the stereo pair is excluded. All are squashed to 256×256 **without preserving aspect ratio**. | `kit_scenes/camera.py:24-40,145`; measured from a raw archive |

## B. Measurements

| Quantity | Value |
|---|---|
| Full KITScenes train split | 2,619 GB, 533 archives, mean 4.91 GB |
| Download throughput | **111 MB/s** — 60 archives in 12.3 min |
| Packing | **~35 s/scene**; 272 GB raw → 381 MB packed (~170×) |
| Peak VRAM (6 cams, bs=1 / 2 / 4) | 5.63 GB / 12.08 GB / **OOM** |
| Training throughput | 10.36 samples/s (residual), 8.57 (deformable) |
| Epoch time, 48 partitions | ~7.5 min |
| Scene eligibility, 4–6 GB archives | 56/60 non-empty; 48 after the navigation audit |
| Repo test suite | 927 passed, 3 failed (all `physical_ai_av`, an optional NVIDIA-only package) |

**Archive size predicts eligibility.** Selecting 4–6 GB archives gave 93%
non-empty; a contributor sampling without that filter reported ~60% discarded.
A scene needs 129 consecutive good frames (12.9 s at 10 Hz) for one sample.

## C. Results

Two runs, 48 partitions, 6 validation scenes, `group_digest 6c5430fa…`, seed 149,
batch 1×4, amp off, subset scope. Both early-stopped at epoch 8, best at epoch 3.

| | residual | deformable |
|---|---|---|
| Best ADE@3s | 1.179 m | **1.053 m** |
| Best FDE@3s | 3.481 m | **2.885 m** |

- **C1 — The A/B difference is inside the noise.** Deformable swings 1.05→2.07 m
  across its own epochs; the gap to residual is 0.13 m. One seed, six validation
  scenes: not a demonstrated effect. Honest reading: deformable is not worse and
  costs 17% throughput.
- **C2 — Both runs overfit after epoch 3**, identically under both fusions:
  training loss 0.26→0.13 while validation ADE flatlines. At 48 scenes the
  limiting factor is data quantity, not architecture. This matches the WG note
  *"network quality depends on input data quality, not just architecture changes"*.
- **C3 — Our number may not beat a trivial baseline.** PR #189 reports
  constant-velocity at **0.883 m ADE@3s** with no perception; our best is 1.053 m.
  Different split, so not directly comparable — but `evaluation/baselines.py`
  states the test: *"if the model barely beats constant-velocity, perception isn't
  contributing"*. **Measure it on our own split before reporting.**

## D. Environment traps

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: lark`, pytest plugin errors | ROS 2 on `PYTHONPATH` → `env -u PYTHONPATH` |
| `pyproj==3.7.2` unsatisfiable | Python < 3.11 |
| `navigation rasterizer shared library is missing` | run `Model/navigation/native/build.py` |
| `validation manifest dataset version does not match` | packed with `DATASET_PACK_VERSION` (v2.2) instead of v3.3 |
| `MlflowException: … maintenance mode` | use `sqlite:///…`, not `file://` |
| `KeyError: 'MLFLOW_TRACKING_URI'` | read as bare `os.environ[...]` |
| STS / checkpoint upload failure | set `AUTO_E2E_CHECKPOINT_BUCKET` + `AWS_ENDPOINT_URL` (disk-backed S3; an in-memory mock dies on ~890 MB checkpoints) |
| CUDA OOM mid-run | a second training process on the same GPU |
| Loss perfectly flat, run looks healthy | `amp=true` — the GradScaler skips every optimizer step |
| Map features have no effect | `lanelet2` missing → map tiles silently become zero tensors |

Also: the dataset is gated — accept the terms **and** use a token with global
gated-repo read (a fine-grained token scoped to your own namespace returns 403).
HuggingFace `main` has moved past the pinned revision `6fde0034…`.
Packing needs `kitscenes` (GitHub only, `--no-deps`) and `lanelet2` (a cp312 wheel
ships inside the SDK repo); the SDK's pinned `opencv<4.10` is built for numpy 1.x
and crashes against the repo's numpy 2.2.6.

## E. Overlap with open PRs — resolve before opening ours

| PR | Author | Overlap |
|---|---|---|
| #172 | riita10069 | **Already adds `planner_mode` to `train_il`** — half of our harness change. Also wires `compute_planner_loss` (fixes A3). |
| #188 | gcordova10 | Adds `camera_bev_size` to the same `train_il` signature block — merge conflict risk; changes A6. |
| #187 | gcordova10 | Packing scripts + `Docs/training_on_a_local_machine.md` — overlaps `Docs/running_experiments.md`. |
| #189 | FLagbusted | Local training/eval scripts + constant-velocity baseline — overlaps `Tools/experiments/`; source of C3. |
| #186 | riita10069 | Large loss-objective PR touching `workflows.py`, adds a `gru` planner. |

**Consequence:** narrow our PR to `map_fusion_mode` only (unique to us), drop the
`planner_mode` half, rebase on `main`, and comment on #172 rather than compete.

## F. Notes on other contributors' reports

- exp-2 (deformable, 50 train scenes) reports **ADE@3s 1.16 m** — consistent with
  our 1.053/1.179, which corroborates the pipeline.
- exp-2 vs exp-3 is **confounded**: exp-3 warm-starts from exp-2's best
  checkpoint, so it has seen more total training. Not a clean comparison.
- Earlier thread numbers (12.79 m, 3.62 m ADE) came from custom scripts off the
  contract, which is consistent with A1/A6 being invisible in a run log.

---

## G. Next actions, in order

1. **Constant-velocity baseline on our `group_digest`** — decides whether §C is a
   result at all. `constant_velocity_baseline()` already exists.
2. **Narrow and rebase the harness PR** (E).
3. **Second seed per fusion** — without it, C1 is not publishable.
4. **More data** (150–200 scenes) — C2 says this is the real lever.
5. **Post the correction** to #168 with the provenance convention (A5), labelled
   as a subset run.
