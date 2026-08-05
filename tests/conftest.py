"""Shared fixtures for the pre-refactor regression tests.

Nothing in this file pulls in the simulator (vmas) or torchrl environment wrappers —
those imports only happen inside individual tests that need them, so unit tests stay
fast and don't require xvfb.
"""
import os
import sys

import pytest
import torch

# Ensure /repo is importable without requiring PYTHONPATH on the command line.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture(scope="session")
def device():
    # Tests default to CPU for portability. GPU-specific tests should use `cuda_device`.
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def small_obs_config():
    """A small, deterministic config that exercises the full obs layout.

    n_worlds and n_agents are intentionally different (10 vs 12) so any test that
    accidentally swaps the "worlds" and "agents" axes in a shape assertion will fail
    loudly — if they were equal, (10, 10, F) would mask bugs in either direction.
    """
    return {
        "n_worlds": 10,
        "n_agents": 12,
        "lidar_history_len": 3,
        "num_lidar_rays": 120,
        "use_global_path": True,
    }


@pytest.fixture
def small_obs_config_no_gp():
    """Same shape as `small_obs_config` but with the global-path block dropped.

    Exists because the model's `combined_dim` calculation branches on the GP flag
    (`LocalNavigationGraphNetDist.__init__` adds +3 when use_global_path_obs=True)
    and the obs layout drops a 5-feature block. With only the GP=True fixture in
    use, a refactor that mis-handles either branch would not be caught. Tests that
    specifically exercise the GP branch point should cover both fixtures; tests
    that are GP-agnostic can stick with `small_obs_config`.
    """
    return {
        "n_worlds": 10,
        "n_agents": 12,
        "lidar_history_len": 3,
        "num_lidar_rays": 120,
        "use_global_path": False,
    }


@pytest.fixture
def small_dynamics():
    return {
        "omega_limit": 1.0,
        "v_limit": 1.0,
        "lidar_obs_range": 3.5,
        "lidar_hist": 3,
        "lidar_rays": 120,
        "agent_radius": 0.2,
    }


def expected_obs_feat(cfg):
    """Total feature-dim of a single per-agent observation with the given config.

    This is the inverse of `observation_to_dict`; if the obs layout ever changes, either
    this helper or the scenario's observation() is out of sync.
    """
    feat = 2 + 6  # goal + state
    if cfg["use_global_path"]:
        feat += 5
    if cfg["n_agents"] > 1:
        feat += 4 * (cfg["n_agents"] - 1)
    feat += cfg["lidar_history_len"] * cfg["num_lidar_rays"]
    return feat


@pytest.fixture
def synthetic_obs(small_obs_config):
    """A deterministic, well-shaped synthetic observation tensor.

    Shape: (n_worlds, n_agents, F). Values are small but nonzero so the GNN's
    "nonzero velocity" filter sees at least some qualifying peers.
    """
    cfg = small_obs_config
    F = expected_obs_feat(cfg)
    torch.manual_seed(0)
    obs = torch.randn(cfg["n_worlds"], cfg["n_agents"], F) * 0.3
    # Force peer velocities to be clearly nonzero so the GNN branch exercises.
    # Layout reminder:
    #   goal(2) | state(6) | global_path(5) | other_agents(4*(N-1)) | lidar(H*R)
    start = 2 + 6 + (5 if cfg["use_global_path"] else 0)
    n_peers = cfg["n_agents"] - 1
    for p in range(n_peers):
        # each peer block = [dx, dy, vx, vy]; bias vx, vy away from 0
        obs[..., start + 4 * p + 2] = 0.5
        obs[..., start + 4 * p + 3] = 0.5
        # keep peers close enough that the distance filter keeps them
        obs[..., start + 4 * p + 0] = 0.1
        obs[..., start + 4 * p + 1] = 0.1
    return obs
