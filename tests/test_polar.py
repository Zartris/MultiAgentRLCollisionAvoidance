"""Polar / angle helper invariants.

to_polar converts (target_pose, agent_pose, agent_rot) into (dist, signed_angle_diff).
The rest of the pipeline relies on (a) distances being nonnegative, (b) the angle
being in [-pi, pi], and (c) the dist being invariant to agent rotation.
"""
import math
import torch

from scenario.CollisionAvoidance_base import angle_diff, angle_to_point, to_polar


def test_to_polar_shape_preserved():
    # to_polar slices along the last axis; any leading shape should pass through
    # unchanged except for the last dim collapsing from (2) xy to (2) (dist, angle).
    # We only test 2D+ inputs because angle_to_point unsqueezes 1D to (1, 2), which
    # breaks the shape round-trip — the real code never calls to_polar with 1D.
    for shape in [(4, 2), (3, 5, 2), (2, 3, 4, 2)]:
        target = torch.randn(*shape)
        agent = torch.randn(*shape)
        rot = torch.randn(*shape[:-1], 1)
        out = to_polar(target, agent, rot)
        assert out.shape == (*shape[:-1], 2)


def test_to_polar_distance_nonnegative():
    torch.manual_seed(0)
    target = torch.randn(100, 2)
    agent = torch.randn(100, 2)
    rot = torch.randn(100, 1)
    out = to_polar(target, agent, rot)
    assert (out[..., 0] >= 0).all(), "polar dist must be >= 0"


def test_to_polar_angle_in_pi_range():
    torch.manual_seed(0)
    target = torch.randn(50, 2)
    agent = torch.randn(50, 2)
    rot = torch.randn(50, 1)
    out = to_polar(target, agent, rot)
    assert (out[..., 1] >= -math.pi - 1e-5).all()
    assert (out[..., 1] <= math.pi + 1e-5).all()


def test_to_polar_dist_independent_of_rotation():
    # Rotating the agent should not change how far away a target is.
    target = torch.tensor([[3.0, 4.0]])
    agent = torch.tensor([[0.0, 0.0]])
    rots = torch.linspace(-math.pi, math.pi, steps=8).unsqueeze(-1)
    dists = []
    for r in rots:
        out = to_polar(target, agent, r.unsqueeze(0))
        dists.append(out[0, 0].item())
    # All eight dists should be 5.0 (the 3-4-5 triangle) regardless of rotation
    for d in dists:
        assert abs(d - 5.0) < 1e-5


def test_to_polar_angle_flips_with_rotation():
    # If target is directly in front (agent facing +x, target at +x), angle_diff ~ 0.
    # If we spin the agent 180 deg, the target is directly behind -> angle_diff ~ +/- pi.
    target = torch.tensor([[1.0, 0.0]])
    agent = torch.tensor([[0.0, 0.0]])
    rot_forward = torch.tensor([[0.0]])  # (1, 1) facing +x
    out = to_polar(target, agent, rot_forward)
    assert abs(out[0, 1].item()) < 1e-4
    rot_backward = torch.tensor([[math.pi]])
    out = to_polar(target, agent, rot_backward)
    # angle diff is +/- pi (sign is implementation-defined at the boundary)
    assert abs(abs(out[0, 1].item()) - math.pi) < 1e-4


def test_angle_diff_wraps_to_pi_range():
    torch.manual_seed(1)
    a = torch.randn(50, 2)
    b = torch.randn(50, 2)
    # Normalize so they're direction vectors
    a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    b = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    diff = angle_diff(a, b)
    assert (diff >= -math.pi - 1e-5).all()
    assert (diff <= math.pi + 1e-5).all()


def test_angle_to_point_returns_unit_vector_by_default():
    # The default return_rads=False path returns a direction vector, not an angle.
    source = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[3.0, 4.0]])
    out = angle_to_point(source, target)
    assert out.shape == (1, 2)
    norm = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)


def test_angle_to_point_rads_in_range():
    torch.manual_seed(0)
    src = torch.randn(100, 2)
    tgt = torch.randn(100, 2)
    a = angle_to_point(src, tgt, return_rads=True)
    assert (a >= -math.pi - 1e-5).all()
    assert (a <= math.pi + 1e-5).all()
