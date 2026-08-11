"""Synthetic forward/backward smoke test for AutoE2E on the KITScenes contract.

Runs the model on random tensors shaped exactly as the packed KITScenes shards
deliver them, so the model, the CUDA build and the GPU can be validated before
any data is downloaded or packed.

Shapes and the BEV grid come from ``DEFAULT_NAVIGATION_GEOMETRY`` rather than
the ``BEVViewFusion`` defaults, because that is what ``train_il`` passes for
KITScenes (``Platform/pipelines/workflows.py:3387``).

Usage:
    python Tools/smoke_forward_pass.py                     # cuda if available
    python Tools/smoke_forward_pass.py --device cpu
    python Tools/smoke_forward_pass.py --no-pretrained     # offline
"""

import argparse
import pathlib
import sys
import time

import torch

_MODEL_DIR = pathlib.Path(__file__).parent.parent.resolve() / "Model"
sys.path.insert(0, str(_MODEL_DIR))

from model_components.auto_e2e import AutoE2E  # noqa: E402
from navigation.geometry import (  # noqa: E402
    DEFAULT_NAVIGATION_GEOMETRY,
    MAP_CHANNEL_COUNT,
    ROUTE_CHANNEL_COUNT,
)
from model_components.losses.trajectory_loss import TrajectoryImitationLoss  # noqa: E402

NUM_VIEWS = 7
IMAGE_SIZE = 256
NUM_TIMESTEPS = 64
NUM_SIGNALS = 2
EGOMOTION_DIM = 256
VISUAL_HISTORY_DIM = 896


def build_batch(batch_size, map_channels, route_channels, device):
    geom = DEFAULT_NAVIGATION_GEOMETRY
    return {
        "camera_tiles": torch.randn(
            batch_size, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE, device=device
        ),
        "map_context": torch.randn(
            batch_size, map_channels, geom.height_px, geom.width_px, device=device
        ),
        "route_mask": torch.randn(
            batch_size, route_channels, geom.height_px, geom.width_px, device=device
        ),
        "visual_history": torch.zeros(batch_size, VISUAL_HISTORY_DIM, device=device),
        "egomotion_history": torch.randn(batch_size, EGOMOTION_DIM, device=device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--backbone", default="swin_v2_tiny")
    parser.add_argument("--map_fusion_mode", default="residual")
    parser.add_argument("--planner_mode", default="bezier")
    # Defaults come from the navigation contract (MapChannel / RouteChannel),
    # not from a hand-copied number: train_il reads the equivalent counts off
    # the packed manifest (workflows.py:3367).
    parser.add_argument("--map_channels", type=int, default=MAP_CHANNEL_COUNT)
    parser.add_argument("--route_channels", type=int, default=ROUTE_CHANNEL_COUNT)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[env] torch {torch.__version__}  device={device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(
            f"[env] {torch.cuda.get_device_name(0)}  "
            f"free={free / 1e9:.1f} GB / total={total / 1e9:.1f} GB"
        )

    geom = DEFAULT_NAVIGATION_GEOMETRY
    print(
        f"[contract] geometry={geom.geometry_id} "
        f"raster={geom.height_px}x{geom.width_px} "
        f"bev={geom.matching_bev_h}x{geom.matching_bev_w}"
    )
    print(
        f"[model] backbone={args.backbone} map_fusion={args.map_fusion_mode} "
        f"planner={args.planner_mode} pretrained={args.pretrained}"
    )

    t0 = time.time()
    model = AutoE2E(
        backbone=args.backbone,
        num_views=NUM_VIEWS,
        embed_dim=256,
        is_pretrained=args.pretrained,
        view_fusion_kwargs=geom.camera_bev_kwargs(),
        map_context_channels=args.map_channels,
        route_channels=args.route_channels,
        map_fusion_mode=args.map_fusion_mode,
        planner_mode=args.planner_mode,
        enable_route_conditioning=True,
        enable_world_model=False,
        enable_reasoning=False,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] built in {time.time() - t0:.1f}s  params={params / 1e6:.1f}M")

    batch = build_batch(args.batch_size, args.map_channels, args.route_channels, device)

    # Inference pass.
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        traj = model(
            batch["camera_tiles"],
            batch["map_context"],
            batch["visual_history"],
            batch["egomotion_history"],
            route_mask=batch["route_mask"],
            geometry_type="pseudo",
            mode="infer",
        )
    print(f"[infer] {time.time() - t0:.2f}s  trajectory={tuple(traj.shape)}")
    expected = (args.batch_size, NUM_TIMESTEPS * NUM_SIGNALS)
    assert tuple(traj.shape) == expected, f"expected {expected}, got {tuple(traj.shape)}"

    # Train step: forward, loss, backward. Proves gradients actually flow.
    model.train()
    target = torch.randn(args.batch_size, NUM_TIMESTEPS, NUM_SIGNALS, device=device)
    loss_fn = TrajectoryImitationLoss(loss_type="smooth_l1").to(device)
    t0 = time.time()
    pred = model(
        batch["camera_tiles"],
        batch["map_context"],
        batch["visual_history"],
        batch["egomotion_history"],
        route_mask=batch["route_mask"],
        geometry_type="pseudo",
        mode="train",
    )
    if isinstance(pred, tuple):
        pred = pred[0]
    loss = loss_fn(pred, target)
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(
        f"[train] {time.time() - t0:.2f}s  loss={loss.item():.4f}  "
        f"grad_norm={grad_norm:.4f}"
    )
    assert torch.isfinite(loss), "loss is not finite"
    assert torch.isfinite(grad_norm) and grad_norm > 0, "no gradient reached the model"

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[mem] peak allocated={peak:.2f} GB at batch_size={args.batch_size}")

    print("\nSMOKE PASS: model builds, runs, and gradients flow.")


if __name__ == "__main__":
    main()
