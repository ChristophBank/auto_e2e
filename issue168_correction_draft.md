# Draft: comment for issue #168

Before we add more runs to this thread, three things about the sanctioned training
path that I think explain most of the incomparability — and one of them means a cell
of the matrix cannot currently be evaluated at all.

## 1. Two of the three declared axes are not reachable through `train_il`

The issue asks us to vary backbone, map-BEV fusion and planner. Only the first is a
`train_il` parameter.

`train_il` builds the model at `Platform/pipelines/workflows.py:3898`:

```python
model = AutoE2E(
    backbone=bb, num_views=num_views, embed_dim=256,
    is_pretrained=True,
    view_fusion_kwargs=view_fusion_kwargs,
    map_context_channels=map_context_channels,
    route_channels=route_channels,
    enable_route_conditioning=enable_route_conditioning,
    enable_reasoning=enable_reasoning, reasoning_mode=reasoning_mode,
    enable_world_model=enable_world_model,
).to(device)
```

Neither `map_fusion_mode` nor `planner_mode` is passed, so both take the
`AutoE2E.__init__` defaults — `map_fusion_mode="residual"`, `planner_mode="bezier"`
(`Model/model_components/auto_e2e.py:28,30`). There is no `train_il` argument for
either.

So every Residual-vs-Attention and Bezier-vs-Flow-Matching number posted in this
thread necessarily came from a custom training script, i.e. off the contract that
`dataset_policy` exists to enforce. That is a bigger source of divergence than the
train/val split, because it is invisible: the run looks like a `train_il` run and
logs like one.

The pieces to fix this are all already in the repo — `BACKBONE_REGISTRY`,
`MAP_FUSION_REGISTRY` (`residual` / `cross_attn` / `deformable`) and
`PLANNER_REGISTRY` (`bezier` / `flow_matching`) are all keyed by string, and
`AutoE2E.__init__` already accepts `map_fusion_mode` / `planner_mode` /
`map_fusion_kwargs` / `planner_kwargs`. What is missing is the pass-through: two
Flyte enums next to the existing `Backbone` enum, two `train_il` parameters, and
forwarding them into the constructor. I am happy to open that PR — it is the
smallest change that makes this issue's matrix actually runnable, and it is
close to what @riita10069 asked for in
https://github.com/autowarefoundation/auto_e2e/issues/168#issuecomment-... ("source
code for exploratory training", not one-off results).

## 2. `flow_matching` is not trainable via the current loop — please do not spend a night on it

`build_planner` raises a `RuntimeWarning` when you select it
(`Model/model_components/trajectory_planning/__init__.py`):

> `planner_mode='flow_matching'` is NOT correctly trainable via the current
> `train_il` loop (it L1-regresses an Euler-from-noise rollout, not the velocity-MSE
> flow objective). Use `'bezier'` unless/until `compute_planner_loss` is wired in.

`train_il` regresses `forward()`'s output with SmoothL1, but `forward()`
Euler-integrates from a fresh noise sample each step, which drives the model to the
conditional mean. The correct objective exists in `compute_planner_loss` but is not
called from the training loop. Until it is, the Flow-Matching column of the matrix
produces numbers that do not mean what they look like. Wiring
`compute_planner_loss` into `train_il` should probably be its own issue.

## 2b. `cross_attn` map fusion is hard-guarded off at the contract BEV resolution

The "Attention" option for map fusion has three registry entries, and only one of
them is usable on the KITScenes contract. `MapCrossAttentionFusion.forward` raises
above 4096 tokens
(`Model/model_components/map_encoder/map_bev_fusion/cross_attention_fusion.py:70`):

```python
if n_tokens > 4096:
    raise ValueError(
        f"cross_attention map fusion is O(N^2) and infeasible at "
        f"{H}x{W}={n_tokens} tokens ..."
    )
```

The contract BEV grid is 256×256 = 65,536 tokens, so `cross_attn` refuses to run
(it raises rather than OOMs — the guard is deliberate). `deformable`
(@intisar1020's #184) is therefore the only attention-based map fusion that is
viable at the pinned resolution, which I think makes it the intended reading of
the "Attention" cell in this issue.

Net: of the 3 × 2 × 2 = 12 cells in the issue's matrix, **6 are trainable as
specified** — {swin, convnext, resnet50} × {residual, deformable} × {bezier} — and
only once the pass-through in point 1 exists. The other six are not "broken" in the
same way, and the difference matters: the `flow_matching` cells *run* fine
(forward, backward and gradients all work) but optimize the wrong objective, so
they yield a number that looks valid and is not; the `cross_attn` cells are refused
outright by the guard above, which is the safer failure.

## 3. The pinned split has moved to v3 — and the digest alone is not enough to prove comparability

@gcordova10's analysis of the frozen split is right, but it cites
`kitscenes_train_dev_v2.json` and that is now stale. `Model/training/dataset_policy.py:121`
resolves **v3**:

```python
validation_split_id="kitscenes_train_dev_v1",
validation_manifest="splits/kitscenes_train_dev_v3.json",
validation_manifest_schema="kitscenes_train_dev_split_v3",
```

All three manifests (v1/v2/v3) carry the same `split_id` (`kitscenes_train_dev_v1`)
and the same `eligible_group_uid_digest`, but different `dataset_version`
(v2.2 / v3.1 / v3.3) and different `packed_contract_digest`
(`a0bf504e…` / `c81a5746…` / `6fb9d857…`).

That matters for the "paste your `group_digest`" proposal: two of us can match the
validation group digest — same 40 scenes, same 3,820 samples — and still have packed
against different contracts, which `validate_*` will reject against each other but
which nothing in a posted metric would reveal. **The reportable tuple is
`group_digest` + `dataset_version` + `packed_contract_digest`**, and it should be
copied out of the run log, not from memory. Anyone who packed before the v3 bump
should expect `validation manifest dataset version does not match packed shards`
rather than a silently different result — which is the good outcome.

## 4. BEV resolution: the contract already pins it, at 256×256

Worth stating because two runs in this thread changed it for memory reasons. For
KITScenes, `train_il` does not use the 450×300 `BEVViewFusion` default — it
overwrites `view_fusion_kwargs` from the navigation geometry contract
(`workflows.py:3387`):

```python
view_fusion_kwargs = DEFAULT_NAVIGATION_GEOMETRY.camera_bev_kwargs()
```

which is `bev_h=256, bev_w=256` (`Model/navigation/geometry.py:225-226`). The 450×300
grid only reaches the L2D path. So the dense-`cross_attn` OOM at 135K tokens and the
drop to 200×200 were both consequences of running outside the KITScenes contract; on
the contract the grid is 65K tokens and fixed for everyone. If we standardise on
`train_il`, BEV resolution stops being a free variable.

## 5. A question for @intisar1020 about the rising train loss

You reported train loss rising (0.228 → 0.414 → 0.444) while ADE improved, with fp16
autocast enabled. `train_il` ships `amp=False` by default, and the comment on that
parameter describes a failure that may be the same one:

> AMP off by default: with fp16 autocast the GradScaler detected inf/nan grads every
> step (fp16 overflow somewhere in the BEV projection / Bezier basis / backbone path)
> and skipped `optimizer.step()` FOREVER — weights never updated, so the trajectory
> loss sat perfectly flat (~2.95) while fp32 learns in one step.

Your loss is not flat, so it is not exactly that, but it would be worth checking how
often your `GradScaler` skipped a step — if it is a large fraction, the effective LR
schedule is not what the config says. Anyone else fitting the model into a small GPU
with fp16 may want to check the same thing before trusting a curve.

---

## What I verified locally

I ran a synthetic forward + backward pass (random tensors on the exact contract
shapes — 7×256×256 cameras, 256×256 raster and BEV from
`DEFAULT_NAVIGATION_GEOMETRY`, 14 map + 2 route channels), asserting trajectory
shape, loss finiteness and a non-zero gradient norm:

| backbone | map fusion | planner | result |
|---|---|---|---|
| swin_v2_tiny | residual | bezier | pass — 73.6M params, grad_norm 0.034 |
| swin_v2_tiny | deformable | bezier | pass — 73.9M params, grad_norm 0.069 |
| swin_v2_tiny | cross_attn | bezier | **refused** — 4096-token guard, 65,536 tokens at contract BEV |
| conv_next_v2_tiny | residual | bezier | pass — 73.9M params, grad_norm 0.063 |
| res_net_50 | residual | bezier | pass — 70.1M params, grad_norm 0.073 |
| swin_v2_tiny | residual | flow_matching | runs, but emits the RuntimeWarning above |

(These are random targets, so the loss values carry no meaning — the check is only
that shapes line up and gradients are finite and non-zero. The case against
`flow_matching` is the warning and the missing `compute_planner_loss` wiring, not
anything measured here.)

All of these had to be constructed by calling `AutoE2E(...)` directly, because
`train_il` exposes no argument for two of the three axes — which is point 1.

## Suggested convention going forward

So that the next batch of numbers is comparable, I would propose every result posted
here carries:

```
scope:              full | subset (+ packed partition count)
group_digest:       <from "Validation split: ... group_digest=..." log line>
dataset_version:    <from the packed manifest>
packed_contract_digest: <from the packed manifest>
backbone / map_fusion_mode / planner_mode
amp:                on | off
effective batch:    batch_size x grad_accum_steps
ADE/FDE @ 3s and @ 6.4s
```

and that anything not produced by `train_il` is labelled as such.
