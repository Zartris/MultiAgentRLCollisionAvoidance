# Tests

Pre-refactor regression suite. Every test here pins a shape contract, construction
invariant, or end-to-end "still runnable" property of the training pipeline. The goal is:
when the codebase is refactored, these tests fail loudly if the refactor silently breaks
tensor semantics, observation layouts, or the env → policy → critic interface.

## Running

```
PYTHONPATH=/repo MPLCONFIGDIR=/tmp/mpl xvfb-run -s "-screen 0 1400x900x24" \
    pytest -q tests/
```

`xvfb-run` is only needed for the env-building tests (`test_env.py`,
`test_rollout.py`) — pyglet requires a display. The pure shape/tensor tests run without
it.

To run only the fast (no-env) tests:

```
PYTHONPATH=/repo pytest -q tests/ -k "not env and not rollout and not sensor"
```

## What's covered

| File                      | Focus                                                               |
|---------------------------|---------------------------------------------------------------------|
| `test_obs_dict.py`        | `observation_to_dict` layout + shapes for every obs-mode combo      |
| `test_polar.py`           | `to_polar`, `angle_to_point`, `angle_diff` shape + sign invariants  |
| `test_normalize.py`       | `normalize_observation` length + per-field division                 |
| `test_prepare_obs.py`     | `LocalNavigationNetBase.prepare_obs` tensor contract                |
| `test_models.py`          | Each LocalNavigation* net: construction + forward shape             |
| `test_multi_agent_net.py` | `MultiAgentLocalNavNet` shared/non-shared × forward shape           |
| `test_action_scaler.py`   | `ActionScaler` output shape + scaling correctness                   |
| `test_graph.py`           | `convert_to_incremental`, `AgentGraphNet` forward                   |
| `test_env.py`             | Every scenario type can be constructed and reset (needs xvfb)       |
| `test_rollout.py`         | `make_network` + one-step env → policy → critic → env (needs xvfb)  |
| `test_ppo_loss.py`        | `ClipPPOLoss` can be built and run on synthetic data                |

## Notes on the test design

- **CPU-first.** Tests default to `cpu` so they run anywhere; CUDA tests are skipped if
  no GPU is present.
- **Synthetic obs.** Most tests build obs tensors from scratch with the exact layout
  documented in `scenario/CollisionAvoidance_base.py:observation_to_dict`. That lets us
  exercise the model code without spinning up the simulator, which is slow.
- **The "batch abuse" assertion.** Several tests verify the flatten/unflatten symmetry
  in the net forwards — if a refactor ever changes how worlds×agents are collapsed, at
  least one of these will fail.
