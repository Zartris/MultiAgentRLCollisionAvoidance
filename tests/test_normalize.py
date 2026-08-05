"""Sanity checks on normalize_observation and the model's .normalize()."""
import torch

from scenario.CollisionAvoidance_base import normalize_observation


def test_normalize_observation_preserves_length():
    # Normalization is element-wise / dim-wise scaling; it must not change the feature count.
    H, R = 3, 120
    F = 7 + H * R  # the normalize_observation layout stops at lidar_dist
    obs = torch.randn(8, F)
    out = normalize_observation(obs, v_limit=1.0, omega_limit=1.0, lidar_obs_range=3.5,
                                lidar_hist=H, lidar_rays=R)
    assert out.shape == (8, F)


def test_normalize_observation_divides_lidar_by_range():
    H, R = 3, 60
    F = 7 + H * R
    obs = torch.zeros(1, F)
    obs[..., 7:] = 3.5  # mark the whole lidar block with the limit
    out = normalize_observation(obs, v_limit=1.0, omega_limit=1.0, lidar_obs_range=3.5,
                                lidar_hist=H, lidar_rays=R)
    assert torch.allclose(out[..., 7:], torch.ones_like(out[..., 7:]))


def test_normalize_observation_scales_angles_by_omega():
    H, R = 3, 10
    F = 7 + H * R
    obs = torch.zeros(1, F)
    # goal_angle at index 1, angle_to_path at 5, angle_to_lookahead at 6
    obs[0, 1] = 2.0
    obs[0, 5] = 4.0
    obs[0, 6] = -6.0
    out = normalize_observation(obs, v_limit=1.0, omega_limit=2.0, lidar_obs_range=1.0,
                                lidar_hist=H, lidar_rays=R)
    assert out[0, 1].item() == 1.0  # 2 / 2
    assert out[0, 5].item() == 2.0  # 4 / 2
    assert out[0, 6].item() == -3.0  # -6 / 2
