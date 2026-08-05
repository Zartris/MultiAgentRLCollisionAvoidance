"""ActionScaler: scales (v, w, *tail) → (v*max_v, w*max_w, *tail)."""
import torch

from models.ActionScaler import ActionScaler


def test_action_scaler_output_shape_matches_input():
    scaler = ActionScaler(max_vel=2.0, max_ang_vel=3.0)
    x = torch.randn(5, 10, 4)  # 4 = v, w, scale_v, scale_w
    out = scaler(x)
    assert out.shape == x.shape


def test_action_scaler_scales_v_w():
    scaler = ActionScaler(max_vel=2.0, max_ang_vel=3.0)
    x = torch.tensor([[1.0, 1.0, 0.1, 0.2]])
    out = scaler(x)
    # Use allclose to tolerate the float32 rounding on 0.1 and 0.2.
    assert torch.allclose(out[0], torch.tensor([2.0, 3.0, 0.1, 0.2]), atol=1e-5)


def test_action_scaler_broadcasts_over_leading_dims():
    scaler = ActionScaler(max_vel=1.5, max_ang_vel=0.5)
    # (worlds, agents, features)
    x = torch.ones(3, 7, 4)
    out = scaler(x)
    assert out[..., 0].allclose(torch.full((3, 7), 1.5))
    assert out[..., 1].allclose(torch.full((3, 7), 0.5))
