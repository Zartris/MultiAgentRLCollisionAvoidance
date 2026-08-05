import torch as th
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal

from models.ActionScaler import ActionScaler
from models.MultiAgentLidarModel import MultiAgentLocalNavNet
from models.baseline.GA3C_CADRL_policy import GA3CPolicy
from models.baseline.net import CNNPolicy
from models.baseline.rvo_policy import RVOPolicy
from models.distributions import NormalParamExtractor


class _FiniteGuard(nn.Module):
    """Replace any non-finite (NaN/inf) net output with 0 before it becomes the
    action distribution's loc/scale. A no-op when the output is finite (so it does
    not change normal training), but it stops a degenerate/early policy from
    emitting NaN actions that crash vmas (``assert not action.isnan().any()``) in
    both data collection and evaluation.
    """

    def forward(self, x):
        return th.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


dist_selection = {
    "IndependentNormal": IndependentNormal,
    "TanhNormal": TanhNormal,
}


def make_network(env, cfg, model_path, load_only_model, device):
    # Make policy:
    dynamics = {
        "omega_limit": cfg["omega_limit"],
        "v_limit": cfg["v_limit"],
        "lidar_obs_range": cfg["lidar_range"],
        "lidar_hist": cfg["lidar_history_len"],
        "lidar_rays": cfg["num_lidar_rays"],
    }
    if cfg.get("model") == "baseline":
        nav_net = CNNPolicy(
            lidar_history_len=cfg["lidar_history_len"],
            lidar_input_dim=cfg["num_lidar_rays"],
            set_gp_as_goal=cfg.get("set_gp_as_goal", False),
            dynamics=dynamics,
            is_critic=False,
            device=device,
        )
        critic_net = CNNPolicy(
            lidar_history_len=cfg["lidar_history_len"],
            lidar_input_dim=cfg["num_lidar_rays"],
            set_gp_as_goal=cfg.get("set_gp_as_goal", False),
            dynamics=dynamics,
            is_critic=True,
            device=device,
        )
    elif cfg.get("model") == "RVO":
        nav_net = RVOPolicy(
            vmas_env=env,
            dt=cfg["dt"],
            max_neighbors=env.n_agents,
            neighbor_dist=cfg["lidar_range"]*2,
            dynamics=dynamics,
        )
        policy_module = TensorDictModule(
            nav_net,
            in_keys=[("agents", "observation")],
            out_keys=[env.action_key],
        )
        return -1, policy_module, None
    elif cfg.get("model") == "GA3CPolicy":
        nav_net = GA3CPolicy(env, dynamics=dynamics,
                             object_vert_inflate_radius=cfg.get("object_vert_inflate_radius", 0.05))
        policy_module = TensorDictModule(
            nav_net,
            in_keys=[("agents", "observation")],
            out_keys=[env.action_key],
        )
        return -1, policy_module, None
    elif cfg.get("model") == "oursD" or cfg.get("model") == "oursGraph":
        nav_net = MultiAgentLocalNavNet(
            base_net=cfg.get("model"),
            lidar_history_len=cfg["lidar_history_len"],
            lidar_input_dim=cfg["num_lidar_rays"],
            state_input_dim=4,
            conv_channels=[32, 32],
            kernel_sizes=[5, 2],
            gnn_emb_size=64,  # 16
            # env.action_spec in torchrl 0.7+ returns the full Composite (not the leaf),
            # so its .shape[-1] is the number of worlds, not the action dim. The leaf
            # BoundedContinuous for the action lives under full_action_spec_unbatched
            # and has shape (n_agents, action_dim); shape[-1] is the per-agent action dim.
            n_agent_outputs=env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            share_params=True,  # all agents share the same network
            use_global_path_obs=cfg.get("use_global_path_obs"),
            set_gp_as_goal=cfg.get("set_gp_as_goal", False),
            dynamics=dynamics,
            device=device,
            std_max=cfg.get("ppo").get("std_max_start", -1.0),  # -1.0 means no max
            std_min=cfg.get("ppo").get("std_min", 1e-7),  # floor on policy std
            gnn_attention=cfg.get("gnn_attention", "emb"),
        )
        critic_net = MultiAgentLocalNavNet(
            base_net=cfg.get("model"),
            lidar_history_len=cfg["lidar_history_len"],
            lidar_input_dim=cfg["num_lidar_rays"],
            state_input_dim=4,
            conv_channels=[32, 32],
            kernel_sizes=[5, 2],
            gnn_emb_size=64,  # 16
            n_agent_outputs=1,  # 1 value per agent
            n_agents=env.n_agents,
            share_params=True,  # all agents share the same network
            use_global_path_obs=cfg.get("use_global_path_obs"),
            set_gp_as_goal=cfg.get("set_gp_as_goal", False),
            dynamics=dynamics,
            device=device,
            gnn_attention=cfg.get("gnn_attention", "emb"),
        )
    else:
        raise ValueError("Invalid model")

    policy_net = nn.Sequential(
        nav_net,
        _FiniteGuard(),
        ActionScaler(
            max_vel=env.scenario.v_limit, max_ang_vel=env.scenario.omega_limit
        ),
        NormalParamExtractor(  # custom NormalParamExtractor as it can take no mapping
            scale_mapping="none", scale_lb=1e-7,  # 1e-4
        ),
        # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    # since the vmas does not suppert different min and max actions settings we are correcting it here.
    # torchrl 0.11 renamed `env.unbatched_action_spec` to `env.full_action_spec_unbatched`.
    # The returned object is a Composite keyed by the env's action_key; semantics unchanged.
    min_action = env.full_action_spec_unbatched[env.action_key].space.low
    min_action[..., 0] = 0
    env.full_action_spec_unbatched[env.action_key].space.low = min_action
    dist_class = dist_selection[cfg["dist_type"]]
    dist_kwargs = None
    if cfg["dist_type"] == "TanhNormal":
        dist_kwargs = {
            "min": env.full_action_spec_unbatched[env.action_key].space.low,
            "max": env.full_action_spec_unbatched[env.action_key].space.high,
        }
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=dist_class,  # IndependentNormal, TanhNormal,
        distribution_kwargs=dist_kwargs,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )  # we'll need the log-prob for the PPO loss

    # Make critic:
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    current_step = -1
    if model_path is not None and cfg.get("model") != "RVO":
        checkpoint = th.load(model_path)
        # Accept both schemas:
        #   new (default from common.save_checkpoint): policy_state_dict / critic_state_dict
        #   legacy (logger.save_checkpoint + common.save_checkpoint(legacy_format=True)): policy / critic
        policy_key = "policy_state_dict" if "policy_state_dict" in checkpoint else "policy"
        critic_key = "critic_state_dict" if "critic_state_dict" in checkpoint else "critic"
        policy.load_state_dict(checkpoint[policy_key])
        critic.load_state_dict(checkpoint[critic_key])
        if load_only_model:
            current_step = checkpoint["step"]

        print(
            f"Loading checkpoint {model_path} completed. Only loading policy {load_only_model}"
        )
    return current_step, policy, critic
