import hashlib
import inspect

import numpy as np
import pytest
import torch

from Platform.pipelines.inference import (
    load_policy,
    noise_from,
    predict_control,
    sha256_file,
    stable_seed64,
)
from Platform.pipelines.overlay_precompute import (
    _BEVActivationRecorder,
    _downsample_features,
    _spatial_feature_deviation,
    infer_loader_controls,
    infer_loader_overlay,
    seed_batch,
)
from training.dataset_policy import KITSCENES_TRAINING_POLICY


def test_sha256_file_streams_expected_digest(tmp_path):
    payload = (b"auto-e2e-overlay" * 4096) + b"tail"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(payload)

    assert sha256_file(checkpoint, chunk_size=97) == hashlib.sha256(payload).hexdigest()


def test_stable_seed_and_noise_are_identity_scoped():
    identity = ("model-sha", "manifest-sha", "l2d-v1-e000001-f000064", 0)
    assert stable_seed64(*identity) == stable_seed64(*identity)
    assert stable_seed64(*identity) != stable_seed64(
        "model-sha", "manifest-sha", "l2d-v1-e000001-f000065", 0
    )

    first = noise_from(*identity, shape=(128,), device="cpu")
    second = noise_from(*identity, shape=(128,), device="cpu")
    other = noise_from(
        "model-sha",
        "manifest-sha",
        "l2d-v1-e000001-f000065",
        0,
        shape=(128,),
        device="cpu",
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


class _FakePlanner:
    num_timesteps = 64
    num_signals = 2


class _FakeReactive:
    TrajectoryPlanner = _FakePlanner()


class _NoiseEchoPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.Reactive_E2E = _FakeReactive()
        self.reset_count = 0
        self.last_egomotion_history = None

    def reset_visual_history(self):
        self.reset_count += 1

    def forward(
        self,
        camera_tiles,
        map_context,
        visual_history,
        egomotion_history,
        *,
        route_mask,
        map_valid,
        route_valid,
        initial_noise,
        **kwargs,
    ):
        self.last_egomotion_history = egomotion_history.detach().clone()
        return initial_noise + self.anchor


class _InPlaceNormalizingPolicy(_NoiseEchoPolicy):
    """A policy that rescales its egomotion input in place, as AutoE2E does.

    Not a hypothetical: this mirrors ``normalize_egomotion``, which divides the
    speed column by 33. Written in place and re-run over one batch, the scaling
    compounds — so the seed fan must hand each iteration its own tensor and v0
    must be read before any of it happens.
    """

    SPEED_SCALE = 33.0
    ACCELERATION_SCALE = 8.0

    def __init__(self):
        super().__init__()
        self.seen_speeds = []

    def forward(self, camera_tiles, map_context, visual_history,
                egomotion_history, **kwargs):
        history = egomotion_history.reshape(-1, 64, 4)
        self.seen_speeds.append(history[:, -1, 0].detach().clone())
        history[:, :, 0] /= self.SPEED_SCALE
        history[:, :, 1] /= self.ACCELERATION_SCALE
        return super().forward(
            camera_tiles, map_context, visual_history, egomotion_history,
            **kwargs,
        )


class _AddFusion(torch.nn.Module):
    def forward(self, image_bev, navigation_bev):
        return image_bev + navigation_bev


class _DiagnosticNavigationEncoder(torch.nn.Module):
    def forward(self, navigation_input):
        map_context = navigation_input[:, :3]
        route = navigation_input[:, 3:5]
        route_effect = torch.cat(
            [route[:, :1], route[:, 1:2], route.sum(dim=1, keepdim=True)],
            dim=1,
        )
        return map_context + route_effect


class _DiagnosticReactive:
    TrajectoryPlanner = _FakePlanner()
    FeatureFusion = torch.nn.Identity()
    NavigationEncoder = _DiagnosticNavigationEncoder()
    MapBEVFusion = _AddFusion()
    route_channels = 2


class _DiagnosticPolicy(_NoiseEchoPolicy):
    def __init__(self):
        super().__init__()
        self.Reactive_E2E = _DiagnosticReactive()

    def forward(
        self,
        camera_tiles,
        map_context,
        visual_history,
        egomotion_history,
        *,
        route_mask,
        route_valid,
        initial_noise,
        **kwargs,
    ):
        image_bev = self.Reactive_E2E.FeatureFusion(camera_tiles[:, 0])
        gated_route = route_mask * route_valid.reshape(-1, 1, 1, 1)
        navigation_bev = self.Reactive_E2E.NavigationEncoder(
            torch.cat([map_context, gated_route], dim=1)
        )
        self.Reactive_E2E.MapBEVFusion(image_bev, navigation_bev)
        return initial_noise + self.anchor


def _batch(size):
    return {
        "visual_tiles": torch.zeros(size, 2, 3, 4, 4),
        "map_context": torch.zeros(size, 3, 4, 4),
        "route_mask": torch.zeros(size, 2, 4, 4),
        "map_valid": torch.ones(size, dtype=torch.bool),
        "route_valid": torch.zeros(size, dtype=torch.bool),
        "visual_history": torch.zeros(size, 896),
        "egomotion_history": torch.zeros(size, 256),
    }


def test_predict_control_is_uid_stable_across_batch_order():
    model = _NoiseEchoPolicy().eval()
    uids = ["l2d-v1-e000001-f000064", "l2d-v1-e000002-f000064"]

    original = predict_control(
        model,
        _batch(2),
        sample_uids=uids,
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
    )
    reordered = predict_control(
        model,
        _batch(2),
        sample_uids=list(reversed(uids)),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
    )

    assert original.shape == (2, 64, 2)
    np.testing.assert_array_equal(original[0], reordered[1])
    np.testing.assert_array_equal(original[1], reordered[0])
    assert model.reset_count == 2


def test_predict_control_rejects_uid_count_mismatch():
    with pytest.raises(ValueError, match="sample_uids length"):
        predict_control(
            _NoiseEchoPolicy(),
            _batch(2),
            sample_uids=["only-one"],
            model_artifact_id="model-sha",
            dataset_manifest_digest="manifest-sha",
        )


def test_infer_loader_controls_emits_seed_fan_and_v0():
    model = _NoiseEchoPolicy().eval()
    first = _batch(2)
    first["sample_uid"] = [
        "l2d-v1-e000001-f000064",
        "l2d-v1-e000001-f000065",
    ]
    first["egomotion_history"][:, -4] = torch.tensor([3.0, 4.0])
    second = _batch(1)
    second["sample_uid"] = ["l2d-v1-e000001-f000066"]
    second["egomotion_history"][:, -4] = 5.0

    class Loader(list):
        projection = None
        geometry_type = "pseudo"

    uids, controls, v0, seeds = infer_loader_controls(
        model,
        Loader([first, second]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1),
        device="cpu",
    )

    assert uids == first["sample_uid"] + second["sample_uid"]
    assert controls.shape == (3, 2, 64, 2)
    np.testing.assert_array_equal(v0, [3.0, 4.0, 5.0])
    assert seeds == (0, 1)
    assert not np.array_equal(controls[:, 0], controls[:, 1])


def test_infer_loader_overlay_isolates_encoder_contributions():
    model = _DiagnosticPolicy().eval()
    batch = _batch(2)
    batch["sample_uid"] = [
        "l2d-v1-e000001-f000064",
        "l2d-v1-e000001-f000065",
    ]
    batch["visual_tiles"][:, 0, 0] = 3.0
    batch["visual_tiles"][:, 0, 1] = 4.0
    batch["map_context"][:, 0] = 1.0
    batch["map_context"][:, 1:] = 2.0
    batch["route_mask"][:, 0] = 1.0
    batch["route_mask"][:, 1] = 2.0
    batch["route_valid"][:] = True

    class Loader(list):
        projection = None
        geometry_type = "pseudo"

    _, controls, _, seeds, heatmaps = infer_loader_overlay(
        model,
        Loader([batch]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1),
        device="cpu",
    )

    assert controls.shape == (2, 2, 64, 2)
    assert seeds == (0, 1)
    assert heatmaps.shape == (2, 6, 32, 32)
    np.testing.assert_array_equal(heatmaps[:, 0], 0.0)
    np.testing.assert_array_equal(heatmaps[:, 1], 0.0)
    np.testing.assert_allclose(heatmaps[:, 2], np.sqrt(14.0 / 3.0))
    np.testing.assert_array_equal(heatmaps[:, 3], 0.0)
    np.testing.assert_allclose(heatmaps[:, 4], np.sqrt(45.0 / 3.0))
    np.testing.assert_array_equal(heatmaps[:, 5], 0.0)


def _speed_batch(size, speed, first_frame=64):
    batch = _batch(size)
    batch["sample_uid"] = [
        f"l2d-v1-e000001-f{first_frame + index:06d}" for index in range(size)
    ]
    batch["egomotion_history"].reshape(size, 64, 4)[:, :, 0] = speed
    return batch


class _Loader(list):
    projection = None
    geometry_type = "pseudo"


def test_seed_fan_gives_every_seed_the_same_egomotion_input():
    """A forward that rescales in place must not leak into the next seed.

    Without a per-seed tensor the second seed reads speed/33, the third
    speed/33**2, and so on. Nothing raises — the controls just come from an
    input that is orders of magnitude too small.
    """
    model = _InPlaceNormalizingPolicy().eval()

    infer_loader_controls(
        model,
        _Loader([_speed_batch(2, 30.0)]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1, 2),
        device="cpu",
    )

    assert len(model.seen_speeds) == 3
    for seen in model.seen_speeds:
        torch.testing.assert_close(seen, torch.full((2,), 30.0))


def test_v0_reports_the_recorded_speed_not_the_rescaled_one():
    """v0 feeds trajectory integration, so it must be the raw ego speed.

    ``batch_to_device`` is a no-op on CPU, so the batch handed to the model can
    be the loader's own tensor. Reading v0 after inference would report
    whatever forward left in it.
    """
    model = _InPlaceNormalizingPolicy().eval()

    _, _, v0, _ = infer_loader_controls(
        model,
        _Loader([
            _speed_batch(2, 30.0),
            _speed_batch(1, 12.0, first_frame=128),
        ]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1),
        device="cpu",
    )

    np.testing.assert_allclose(v0, [30.0, 30.0, 12.0])


def test_inference_leaves_the_loader_batch_untouched():
    model = _InPlaceNormalizingPolicy().eval()
    batch = _speed_batch(2, 30.0)
    before = batch["egomotion_history"].clone()

    infer_loader_controls(
        model,
        _Loader([batch]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        base_seeds=(0, 1),
        device="cpu",
    )

    assert torch.equal(batch["egomotion_history"], before)


def test_seed_batch_shares_the_image_tensors_it_does_not_clone():
    """The ego vector is cloned; the ~40 MB image tensors must not be."""
    batch = _speed_batch(2, 30.0)

    seeded = seed_batch(batch)

    assert seeded["egomotion_history"] is not batch["egomotion_history"]
    assert torch.equal(seeded["egomotion_history"], batch["egomotion_history"])
    assert seeded["visual_tiles"] is batch["visual_tiles"]
    assert seeded["map_context"] is batch["map_context"]


def test_spatial_feature_deviation_preserves_channel_direction_changes():
    features = torch.tensor(
        [
            [
                [[1.0, -1.0], [0.0, 0.0]],
                [[-1.0, 1.0], [0.0, 0.0]],
            ]
        ]
    )

    heatmap = _spatial_feature_deviation(
        features,
        preserve_zero_cells=True,
    )

    np.testing.assert_allclose(
        heatmap,
        [[[1.0, 1.0], [0.0, 0.0]]],
    )


def test_spatial_feature_deviation_uses_each_channel_spatial_mean():
    features = torch.tensor(
        [[[[1.0, 3.0], [5.0, 7.0]]]]
    )

    heatmap = _spatial_feature_deviation(features)

    np.testing.assert_allclose(
        heatmap,
        [[[3.0, 1.0], [1.0, 3.0]]],
    )


def test_activation_recorder_rejects_incompatible_models_and_outputs():
    with pytest.raises(ValueError, match="does not expose"):
        _BEVActivationRecorder(torch.nn.Identity())
    with pytest.raises(ValueError, match="must be a"):
        _downsample_features(torch.zeros(1, 2, 3), "image")

    recorder = _BEVActivationRecorder(_DiagnosticPolicy())
    try:
        with pytest.raises(RuntimeError, match="hooks did not run"):
            recorder.take(_batch(1), 1)
    finally:
        recorder.close()


def test_overlay_inference_applies_checkpoint_data_sanitization():
    model = _NoiseEchoPolicy().eval()
    batch = _batch(1)
    batch["sample_uid"] = ["kitscenes-v1-scene-a-f000064"]
    history = batch["egomotion_history"].reshape(1, 64, 4)
    history[:] = 1.0
    history[:, -1, 0] = 3.0

    class Loader(list):
        projection = None
        geometry_type = "pinhole"

    _, _, v0, _ = infer_loader_controls(
        model,
        Loader([batch]),
        model_artifact_id="model-sha",
        dataset_manifest_digest="manifest-sha",
        device="cpu",
        training_policy=KITSCENES_TRAINING_POLICY,
    )

    adapted = model.last_egomotion_history.reshape(1, 64, 4)
    assert torch.count_nonzero(adapted[:, :24]) == 24 * 4
    assert adapted[0, -1, 0].item() == 3.0
    assert adapted[0, -1, 1].item() == 0.0
    np.testing.assert_array_equal(v0, [3.0])


def test_overlay_task_never_decodes_future_images():
    pytest.importorskip("flytekit")
    from Platform.pipelines.overlay_tasks import precompute_overlay_partition

    source = inspect.getsource(
        precompute_overlay_partition.task_function
    )
    assert "decode_future_frames=False" in source


def test_load_policy_filters_config_and_returns_checkpoint_identity(
    tmp_path, monkeypatch
):
    from model_components import auto_e2e as auto_e2e_module

    class TinyPolicy(torch.nn.Module):
        def __init__(self, width=2):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(width))

    source = TinyPolicy(width=3)
    source.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0]))
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"width": 3, "removed_argument": "ignored"},
            "epoch": 1,
        },
        checkpoint,
    )
    monkeypatch.setattr(auto_e2e_module, "AutoE2E", TinyPolicy)

    loaded, config, artifact_id = load_policy(checkpoint, "cpu")

    assert isinstance(loaded, TinyPolicy)
    assert loaded.training is False
    assert torch.equal(loaded.weight, source.weight)
    assert config["removed_argument"] == "ignored"
    assert artifact_id == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
