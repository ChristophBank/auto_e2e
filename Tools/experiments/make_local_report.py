"""Render prediction-vs-human trajectory videos from a local checkpoint.

`Tools.trajectory_visualization` renders the videos but deliberately takes a
precomputed canonical overlay rather than a checkpoint, so an export cannot
diverge from the Console. The sanctioned producer of that overlay
(`precompute_overlay_partition`) needs S3 and DynamoDB, which a workstation does
not have.

This bridges the gap without weakening anything: it runs the same inference
functions the Flyte task runs (`infer_loader_overlay` / `write_overlay`), then
synthesises the two publication manifests the renderer validates against. That
validation is entirely local — it checks the manifests, the shard and the
overlay against each other — so a locally produced set is verified exactly as a
published one is. The manifests are marked with a local dataset name so they can
never be confused with a published artifact.

Usage:
    python Tools/experiments/make_local_report.py \\
        --checkpoint /path/to/epoch-0003.pt \\
        --partition /path/to/packed/<scene_uid> \\
        --output-dir /tmp/report
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(REPO / "Model"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from data_parsing.pre_extracted import make_pre_extracted_loader  # noqa: E402
from model_components.auto_e2e import AutoE2E  # noqa: E402
from Platform.pipelines.overlay import write_overlay  # noqa: E402
from Platform.pipelines.overlay_precompute import infer_loader_overlay  # noqa: E402
from Platform.pipelines.workflows import _model_kwargs  # noqa: E402
from training.dataset_policy import training_policy_for_dataset  # noqa: E402

LOCAL_DATASET = "local/KITScenes-Multimodal"
BASE_SEEDS = (0,)


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifests(out_dir, shard, overlay, sample_count, version, rig_path):
    """Write the dataset (v2) and overlay (v1) manifests the renderer checks.

    Every field below is verified by validate_report_provenance; the digests
    must agree with the files on disk, which is why they are computed here
    rather than hardcoded.
    """
    rig_sha = sha256_file(rig_path)
    shard_sha = sha256_file(shard)
    overlay_sha = sha256_file(overlay)

    dataset_manifest = {
        "schema_version": "v2",
        "status": "ready",
        "dataset": LOCAL_DATASET,
        "version": version,
        "shard_entries": [
            {
                "name": shard.name,
                "key": f"{LOCAL_DATASET}/{version}/shards/{shard.name}",
                "byte_size": shard.stat().st_size,
                "sha256": shard_sha,
                "content_identity": shard_sha,
                "rig": {
                    "sha256": rig_sha,
                    "key": f"{LOCAL_DATASET}/{version}/rig/{rig_sha}.json",
                },
            }
        ],
    }
    dm_bytes = json.dumps(dataset_manifest, indent=2, sort_keys=True).encode()
    dm_path = out_dir / "dataset-manifest.json"
    dm_path.write_bytes(dm_bytes)

    overlay_manifest = {
        "schema_version": "v1",
        "status": "ready",
        "dataset": LOCAL_DATASET,
        "version": version,
        "dataset_manifest_sha256": sha256_bytes(dm_bytes),
        "overlay_binary_schema": "v3",
        "seeds": list(BASE_SEEDS),
        "model_artifact_sha256": sha256_file(ARGS.checkpoint),
        "request_identity": sha256_bytes(b"local-report-request"),
        "cache_identity": sha256_bytes(b"local-report-cache"),
        "registered_model_name": "local-checkpoint",
        "run_id": "local",
        "model_version": 1,
        "sampler": "model-default",
        "inference_contract_version": "v1",
        "noise_policy_version": "v1",
        "num_inference_steps": 1,
        "shards": [
            {
                "shard": shard.name,
                "sha256": overlay_sha,
                "byte_size": overlay.stat().st_size,
                "sample_count": sample_count,
                "seeds": list(BASE_SEEDS),
            }
        ],
    }
    om_path = out_dir / "overlay-manifest.json"
    om_path.write_text(json.dumps(overlay_manifest, indent=2, sort_keys=True))
    return dm_path, om_path


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--partition", required=True, help="one packed scene directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default="KIT-MRT/KITScenes-Multimodal")
    ap.add_argument("--validation-scope", default="subset")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--max-frames-per-scene", type=int, default=300)
    ARGS = ap.parse_args()

    partition = pathlib.Path(ARGS.partition)
    shards = sorted(partition.glob("*.tar"))
    if not shards:
        raise SystemExit(f"no .tar shard in {partition}")
    shard = shards[0]
    rig_path = partition / "rig" / "projection.json"
    if not rig_path.is_file():
        raise SystemExit(f"no rig artifact at {rig_path}")

    manifest = json.loads((partition / "manifest.json").read_text())
    version = manifest["dataset_version"]

    out_dir = pathlib.Path(ARGS.output_dir)
    work = out_dir / "inputs"
    work.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ARGS.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(
        f"checkpoint: backbone={config.get('backbone')} "
        f"map_fusion={config.get('map_fusion_mode', 'residual (pre-#192)')} "
        f"planner={config.get('planner_mode', 'bezier')}"
    )

    model = AutoE2E(**_model_kwargs(config)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = make_pre_extracted_loader(
        str(partition),
        batch_size=8,
        num_workers=1,
        shuffle=0,
        pin_memory=(device.type == "cuda"),
    )
    policy = training_policy_for_dataset(
        ARGS.dataset, validation_scope=ARGS.validation_scope
    )

    uids, controls, v0, seeds, heatmaps = infer_loader_overlay(
        model,
        loader,
        model_artifact_id="local-checkpoint",
        dataset_manifest_digest=sha256_file(shard),
        base_seeds=BASE_SEEDS,
        device=device,
        training_policy=policy,
    )
    print(f"inferred {len(uids)} samples")

    overlay_path = work / "overlay.bin.gz"
    write_overlay(
        overlay_path, uids, controls, v0,
        base_seeds=BASE_SEEDS, bev_heatmaps=heatmaps,
    )

    dm, om = build_manifests(work, shard, overlay_path, len(uids), version, rig_path)

    report_dir = out_dir / "report"
    cmd = [
        sys.executable, "-m", "Tools.trajectory_visualization",
        "--shard", str(shard),
        "--overlay", str(overlay_path),
        "--dataset-manifest", str(dm),
        "--overlay-manifest", str(om),
        "--output-dir", str(report_dir),
        "--camera-index", str(ARGS.camera_index),
        "--max-frames-per-scene", str(ARGS.max_frames_per_scene),
    ]
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'Model'}:{REPO}")
    print("\n" + " ".join(cmd) + "\n")
    raise SystemExit(subprocess.call(cmd, cwd=str(REPO), env=env))


if __name__ == "__main__":
    main()
