"""Run one experiment from experiments.yaml through the sanctioned task path.

Chains audit_kitscenes_navigation_quality -> train_il. There is deliberately no
training loop here: numbers produced by a custom loop are not comparable with
anyone else's, which is the problem this tooling exists to solve.

Local execution needs three things Flyte normally supplies:

    AUTO_E2E_CHECKPOINT_BUCKET  skips the STS get_caller_identity lookup
    AWS_ENDPOINT_URL            points boto3 at a local S3 (MinIO)
    MLFLOW_TRACKING_URI         read as bare os.environ[...], no default

Defaults below assume the MinIO container from Docs/running_experiments.md.

Usage:
    python Tools/experiments/run_experiment.py --run ab_residual --packed /path/to/packed
    python Tools/experiments/run_experiment.py --list
"""

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).resolve().parent / "experiments.yaml"

os.environ.setdefault("AUTO_E2E_CHECKPOINT_BUCKET", "autoe2e-checkpoints")
os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "autoe2e")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "autoe2e123")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(REPO / "Model"))
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402
from flytekit.types.directory import FlyteDirectory  # noqa: E402

from Platform.pipelines.workflows import (  # noqa: E402
    Backbone,
    Dataset,
    MapFusion,
    Planner,
    audit_kitscenes_navigation_quality,
    train_il,
)


def load_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    runs = {r["name"]: r for r in cfg["runs"]}
    return cfg["shared"], runs


def install_camera_ablation():
    """Replace the camera tensor with zeros on every forward.

    Patched at ``AutoE2E.forward`` rather than in the loader because ``train_il``
    builds its own loaders for training and for evaluation. One patch point
    covers both, which is the property that makes this an ablation: zeroing the
    cameras only during training would measure a train/test mismatch instead of
    what the cameras contribute.

    Zero is post-normalization, so the model sees the dataset mean image rather
    than black — the usual "no information, same statistics" ablation.

    Deliberately local. It leaves no mark in ``checkpoint_config``, so an
    ablation checkpoint is indistinguishable from a normal one after the fact;
    the run name and the result JSON are the only record. Do not evaluate one of
    these checkpoints outside this script and report the number as a plain run.
    """
    import torch
    from model_components.auto_e2e import AutoE2E

    original_forward = AutoE2E.forward

    def forward_without_cameras(self, camera_tiles, *args, **kwargs):
        return original_forward(
            self, torch.zeros_like(camera_tiles), *args, **kwargs
        )

    AutoE2E.forward = forward_without_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run name from experiments.yaml")
    ap.add_argument("--packed", help="directory of packed WebDataset partitions")
    ap.add_argument("--results", default=None, help="where to write the result JSON")
    ap.add_argument("--list", action="store_true", help="list configured runs and exit")
    args = ap.parse_args()

    shared, runs = load_config()

    if args.list:
        print(
            f"{'run':<28} {'backbone':<18} {'map_fusion':<12} "
            f"{'planner':<10} {'seed':<6} cameras"
        )
        for name, r in runs.items():
            cameras = "ZEROED" if r.get("zero_cameras") else "on"
            print(
                f"{name:<28} {r['backbone']:<18} {r['map_fusion_mode']:<12} "
                f"{r['planner_mode']:<10} {r['seed']:<6} {cameras}"
            )
        return

    if not args.run or not args.packed:
        ap.error("--run and --packed are required unless --list is given")
    if args.run not in runs:
        ap.error(f"unknown run {args.run!r}; known: {sorted(runs)}")

    run = runs[args.run]
    packed = pathlib.Path(args.packed)
    partitions = sorted(p for p in packed.iterdir() if (p / "manifest.json").is_file())
    if not partitions:
        raise SystemExit(f"no packed partitions under {packed}")

    shards = [FlyteDirectory(path=str(p)) for p in partitions]
    print(f"=== {run['name']} ===")
    print(f"partitions      {len(partitions)}")
    print(
        f"backbone        {run['backbone']}\n"
        f"map_fusion_mode {run['map_fusion_mode']}\n"
        f"planner_mode    {run['planner_mode']}\n"
        f"seed            {run['seed']}"
    )
    zero_cameras = bool(run.get("zero_cameras", False))
    print(f"cameras         {'ZEROED (ablation)' if zero_cameras else 'on'}")
    print(
        f"epochs {shared['epochs']}  batch {shared['batch_size']}"
        f"x{shared['grad_accum_steps']}  lr {shared['lr']}  amp {shared['amp']}\n"
    )
    if zero_cameras:
        install_camera_ablation()
        print(
            "!! CAMERA ABLATION ACTIVE — visual_tiles are zeroed in training "
            "AND evaluation.\n"
            "!! The checkpoint does NOT record this. Its only record is this "
            "run's name and result JSON.\n"
        )

    audit = audit_kitscenes_navigation_quality.task_function(shards=shards)

    t0 = time.time()
    out = train_il.task_function(
        shards=shards,
        dataset=Dataset[shared["dataset"]],
        backbone=Backbone(run["backbone"]),
        map_fusion_mode=MapFusion(run["map_fusion_mode"]),
        planner_mode=Planner(run["planner_mode"]),
        epochs=shared["epochs"],
        batch_size=shared["batch_size"],
        grad_accum_steps=shared["grad_accum_steps"],
        lr=float(shared["lr"]),
        training_seed=run["seed"],
        amp=shared["amp"],
        num_workers=shared["num_workers"],
        val_fraction=shared["val_fraction"],
        validation_scope=shared["validation_scope"],
        navigation_quality_audit=audit,
    )
    elapsed = time.time() - t0

    results_dir = pathlib.Path(args.results or (packed.parent / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **{k: v for k, v in run.items()},
        **{k: v for k, v in shared.items()},
        "partitions": len(partitions),
        "zero_cameras": zero_cameras,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(out.checkpoint),
        "metadata": str(out.metadata),
    }
    (results_dir / f"{run['name']}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n=== {run['name']} finished in {elapsed / 60:.1f} min ===")


if __name__ == "__main__":
    main()
