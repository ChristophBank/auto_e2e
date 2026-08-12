"""Score training-free baselines on the exact validation split of a finished run.

Answers the question `Model/evaluation/baselines.py` poses: does the trained
model beat ego-status extrapolation at all? If it barely does, perception is not
contributing yet.

Comparability is enforced, not assumed. The validation group UIDs are read from
the run's own metadata.json and passed to the same loader factory `train_il`
uses, and scoring goes through the same `_evaluate_open_loop`. The script then
checks that the resulting `sample_uid_digest` equals the one the training run
recorded — if it differs, the numbers are not on the same samples and the script
says so rather than printing a misleading comparison.

The baseline is expressed as a stub module in the model's place, so the
prediction takes the identical integration and metric path as a real checkpoint.

Usage:
    python Tools/experiments/run_baseline.py \
        --metadata /path/to/metadata.json --packed /path/to/packed
"""

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

os.environ.setdefault("AUTO_E2E_CHECKPOINT_BUCKET", "autoe2e-checkpoints")
os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "autoe2e")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "autoe2e123")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

sys.path.insert(0, str(REPO / "Model"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from data_parsing.pre_extracted import make_multi_dataset_loader  # noqa: E402
from navigation.geometry import DEFAULT_NAVIGATION_GEOMETRY  # noqa: E402
from Platform.pipelines.workflows import _evaluate_open_loop  # noqa: E402
from training.dataset_policy import training_policy_for_dataset  # noqa: E402

AUTO_E2E_TIMESTEPS = 64
NUM_SIGNALS = 2


class ConstantVelocity(nn.Module):
    """accel = 0, curv = 0 — maintain speed, drive straight.

    Mirrors evaluation.baselines.constant_velocity_baseline in the module shape
    _evaluate_open_loop expects, so the zeros travel the same integration path
    as a checkpoint's output.
    """

    def forward(self, visual, map_context, visual_history, egomotion_history, **kw):
        b = visual.shape[0]
        return torch.zeros(
            b, AUTO_E2E_TIMESTEPS * NUM_SIGNALS,
            device=visual.device, dtype=torch.float32,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="metadata.json of a finished run")
    ap.add_argument("--packed", required=True, help="directory of packed partitions")
    ap.add_argument("--dataset", default="KIT-MRT/KITScenes-Multimodal")
    ap.add_argument("--validation-scope", default="subset")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    meta = json.loads(pathlib.Path(args.metadata).read_text())
    val = meta["validation"]
    group_uids = list(val["validation_group_uids"])
    expected_sample_digest = val["validation_sample_uid_digest"]

    print(f"validation groups   {len(group_uids)}")
    print(f"group_digest        {val['validation_group_uid_digest']}")
    print(f"expected samples    {val['validation_sample_count']}\n")

    packed = pathlib.Path(args.packed)
    # Skip empty partitions the way train_il does (its skipped_empty count):
    # a scene shorter than 129 usable frames packs a manifest but no shard, and
    # the loader raises FileNotFoundError on the missing .tar.
    shard_dirs = []
    skipped_empty = 0
    for p in sorted(packed.iterdir()):
        manifest = p / "manifest.json"
        if not manifest.is_file():
            continue
        if json.loads(manifest.read_text()).get("total_samples", 0) <= 0:
            skipped_empty += 1
            continue
        shard_dirs.append(str(p))
    print(f"partitions          {len(shard_dirs)} (skipped_empty={skipped_empty})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_policy = training_policy_for_dataset(
        args.dataset, validation_scope=args.validation_scope
    )

    loader = make_multi_dataset_loader(
        shard_dirs,
        batch_size=8,
        num_workers=1,
        shuffle=0,
        pin_memory=(device.type == "cuda"),
        split="val",
        val_fraction=float(val["val_fraction"]),
        validation_group_uids=group_uids,
        max_active_loaders=1,
        prefetch_factor=1,
        decode_future_frames=False,
    )

    result = _evaluate_open_loop(
        ConstantVelocity().to(device),
        loader,
        device,
        training_policy=training_policy,
        navigation_geometry=DEFAULT_NAVIGATION_GEOMETRY,
    )

    same = result["sample_uid_digest"] == expected_sample_digest
    print(f"samples scored      {result['sample_count']}")
    print(f"sample_uid_digest   {result['sample_uid_digest']}")
    print(f"matches the run     {'YES' if same else 'NO'}\n")

    print("constant-velocity baseline (accel=0, curv=0)")
    for label, h in result["horizons"].items():
        print(f"  ADE@{label:<3} {h['ade']:.4f} m    FDE@{label:<3} {h['fde']:.4f} m")

    if not same:
        print(
            "\nWARNING: the scored samples differ from the training run's. "
            "These numbers are NOT comparable with that run's ADE/FDE."
        )

    payload = {
        "baseline": "constant_velocity",
        "sample_count": result["sample_count"],
        "sample_uid_digest": result["sample_uid_digest"],
        "matches_run_validation": same,
        "group_digest": val["validation_group_uid_digest"],
        "ade_3s": result["ade"],
        "fde_3s": result["fde"],
        "horizons": result["horizons"],
    }
    out = pathlib.Path(args.output or (packed.parent / "results" / "baseline_constant_velocity.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
