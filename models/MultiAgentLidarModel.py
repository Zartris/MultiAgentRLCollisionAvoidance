import abc
import typing

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchrl.data import DEVICE_TYPING
from torch_geometric.data import Data, Batch
from torch_geometric.utils import mask_to_index
from torch_geometric.nn import global_mean_pool, global_add_pool
from torchrl.modules.models.utils import _reset_parameters_recursive
from typing_extensions import deprecated

from scenario.CollisionAvoidance_base import (
    normalize_observation,
    observation_to_dict,
    to_polar,
)
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import GATConv

# Profiling decorator is an IDE-only dev dep. Fall back to a no-op if the package
# isn't installed so the module still imports in a bare container.
try:
    from line_profiler_pycharm import profile
except ImportError:
    def profile(fn):
        return fn


def convert_to_incremental(tensor):
    """
    Converts a tensor with batch values to an incremental tensor starting from 0.

    Args:
        tensor (torch.Tensor): The input tensor with batch values.

    Returns:
        torch.Tensor: The converted tensor with incremental values starting from 0.
    """
    # Get the unique values and their inverse indices
    unique_values, inverse_indices = torch.unique(
        tensor, sorted=True, return_inverse=True
    )

    return inverse_indices


class AttentionNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionNet, self).__init__()
        self.lin = torch.nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.lin(x)


class AgentGraphNet(nn.Module):
    """Peer aggregation for the agent graph.

    attention_mode:
      "none" -- legacy: the gate is softmaxed over a size-1 dim, so it is
                identically 1.0 and the module is a plain sum-pool of encoded peers
                (the long-standing no-op). Default, so old checkpoints load and
                reproduce exactly.
      "raw"  -- working attention: gate = AttentionNet(raw 4-d peer features),
                softmax OVER PEERS (masked), weighted sum.
      "emb"  -- working attention, AttentionalAggregation-style: gate =
                AttentionNet(encoded node embedding), softmax OVER PEERS.
    "raw"/"emb" change behaviour vs old checkpoints (intended: retrain with them).
    """

    def __init__(self, in_channels, out_channels, pooling="sum", attention_mode="none"):
        super(AgentGraphNet, self).__init__()
        self.attention_mode = attention_mode
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )
        # Gate on raw peer features. Present in every mode so old checkpoints
        # (node_encoder + node_attention only) still load strictly.
        self.node_attention = AttentionNet(in_channels)
        # "emb" mode adds a gate on the encoded embedding (new params -> only for
        # fresh training, never present in old checkpoints).
        if attention_mode == "emb":
            self.node_attention_emb = AttentionNet(out_channels)

    def forward(self, x, batch):
        gate = self.node_attention(x)
        node_emb = self.node_encoder(x)

        a_x = node_emb * F.softmax(gate, dim=-1)
        graph_emb = global_add_pool(a_x, batch)  # global_add_pool
        return graph_emb

    def forward_dense(self, x, mask):
        """Dense, sync-free peer aggregation. x:[B, P, in], mask:[B, P] bool ->
        [B, out]. Avoids the data-dependent argwhere()/unique() that forced a
        device->host sync every forward.

        "none" reproduces the old sparse graph + global_add_pool exactly (a sum
        over each ego's masked peers; per-node encoders have no cross-node
        interaction). "raw"/"emb" softmax the gate over the peer axis (masked) for
        real attention; egos with no qualifying peers produce a zero embedding.
        """
        node_emb = self.node_encoder(x)                       # [B, P, out]
        keep = mask.unsqueeze(-1)                             # [B, P, 1]
        if self.attention_mode == "none":
            a_x = node_emb * F.softmax(self.node_attention(x), dim=-1)  # *1.0 (no-op)
            return (a_x * keep.to(a_x.dtype)).sum(dim=-2)
        if self.attention_mode == "emb":
            gate = self.node_attention_emb(node_emb)          # [B, P, 1]
        else:  # "raw"
            gate = self.node_attention(x)                     # [B, P, 1]
        gate = gate.masked_fill(~keep, float("-inf"))         # excluded peers -> 0 weight
        attn = F.softmax(gate, dim=-2)                        # over the peer axis
        attn = torch.nan_to_num(attn, nan=0.0)               # egos with no peers -> 0
        return (node_emb * attn).sum(dim=-2)                  # [B, out]


class LIDARDistEncoder(nn.Module):
    def __init__(
        self, input_dim, input_channels, conv_channels, kernel_sizes, output_dim
    ):
        super(LIDARDistEncoder, self).__init__()
        layers = []
        for i in range(len(conv_channels)):
            layers.append(
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else conv_channels[i - 1],
                    out_channels=conv_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=2,
                )
            )
            layers.append(nn.LeakyReLU(0.01))
        self.wrap_featres = 4
        self.conv = nn.Sequential(*layers)
        dummy = torch.zeros(1, input_channels, input_dim + self.wrap_featres)
        dummy_out = self.conv(dummy)
        self.maxpool = nn.MaxPool1d(3, 2)
        dummy_out = self.maxpool(dummy_out)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.flatten(dummy_out).shape[-1], output_dim)
        debug = 0

    def forward(self, x):
        x = torch.cat((x, x[..., : self.wrap_featres]), dim=-1)
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class LocalNavigationNetBase(nn.Module):
    def __init__(
        self,
        lidar_history_len,
        lidar_num_rays,
        state_input_dim,
        conv_channels,
        kernel_sizes,
        output_dim,
        dynamics: typing.Dict,
        use_global_path_obs=False,
        set_gp_as_goal=False,
        device="cpu",
        std_max=-1,
        std_min=1e-7,
    ):
        super().__init__()
        self.lidar_history_len = lidar_history_len
        self.lidar_num_rays = lidar_num_rays
        self.lidar_input_dim = (
            lidar_history_len,
            lidar_num_rays,
        )
        self.lidar_channels = 3  # dist, vel_x, vel_y
        self.state_input_dim = state_input_dim
        self.is_critic = output_dim == 1
        self.use_global_path_obs = use_global_path_obs
        self.set_gp_as_goal = set_gp_as_goal
        self.dynamics = dynamics
        self.device = device
        self.std_max = std_max
        self.std_min = std_min

    def normalize(self, lidar_dist, state_data, other_agents_graph=None):
        omega_limit = self.dynamics.get("omega_limit")
        v_limit = self.dynamics.get("v_limit")
        lidar_obs_range = self.dynamics.get("lidar_obs_range")

        goal_dist = state_data[..., None, 0]
        goal_angle = state_data[..., None, 1] / omega_limit
        agent_vel = state_data[..., None, 2] / v_limit
        agent_ang_vel = state_data[..., None, 3] / omega_limit
        norm_state = torch.cat(
            (goal_dist, goal_angle, agent_vel, agent_ang_vel), dim=-1
        )
        if self.use_global_path_obs:
            gp_dist = state_data[..., None, 4]
            angle_to_point_diff = state_data[..., None, 5] / omega_limit
            angle_diff_to_next = state_data[..., None, 6] / omega_limit
            norm_state = torch.cat(
                (norm_state, gp_dist, angle_to_point_diff, angle_diff_to_next), dim=-1
            )

        lidar_dist = lidar_dist / lidar_obs_range

        if other_agents_graph is not None:
            # dense (peers [B,P,4], mask [B,P]) -- normalize the peer features
            peers, mask = other_agents_graph
            peers = peers.clone()
            peers[..., 0] = peers[..., 0] / lidar_obs_range
            peers[..., 1] = peers[..., 1] / omega_limit
            peers[..., 2:] = peers[..., 2:] / v_limit
            other_agents_graph = (peers, mask)

        return lidar_dist, norm_state, other_agents_graph

    @profile
    def prepare_obs(self, obs, num_agents, use_other_agents=True):
        """Split the per-agent obs into semantic tensors and normalize.

        Shape contract:
            obs arrives pre-flattened to (B, F), where B = n_worlds * n_agents and F is
            the per-agent feature length. The flatten happens in the model's forward()
            via obs.flatten(0, len(batch) - 1), so every tensor in obs_dict has a
            single leading batch axis of size B. The GNN branch below reuses B as the
            "ego-agent node id" when building torch_geometric's graph.batch vector —
            that's the whole reason the caller collapses worlds/agents into one axis
            before we get here.

        Concrete example from LidarSingleStep with W=10 worlds, N=12 agents, H=3
        lidar-history frames, R=120 rays, use_global_path_obs=True:
            F = 2 + 6 + 5 + 4*(N-1) + H*R = 2 + 6 + 5 + 44 + 360 = 417
            obs:                       (10, 12, 417) outside; here already (B=120, 417)
            obs_dict["goal"]           (120, 2)
            obs_dict["state"]          (120, 6)           pos_xy, rot, vel_xy, ang_vel
            obs_dict["global_path"]    (120, 5)           gp_pt_xy, gp_dist, angles
            obs_dict["other_agents"]   (120, 11, 4)       (B, N-1, 4) dx,dy,vx,vy
            obs_dict["lidar_dist"]     (120, 3, 120)      (B, H, R)
        Return:
            lidar_dist      (120, 3, 120)
            state_data      (120, 7)   goal_polar(2) + vel_mag(1) + ang_vel(1) + gp(3)
            graph           PyG Data with x:(num_valid_peers, 4), batch:(num_valid_peers,)
                            and global_bat:(num_egos_with_peers,). Egos with zero peers
                            are missing from global_bat; the caller zero-fills their row.
        """
        batch_size = obs.shape[0]
        obs_dict = self.obs_to_dict(
            obs,
            True,
            self.lidar_history_len,
            self.lidar_num_rays,
            use_global_path=self.use_global_path_obs or self.set_gp_as_goal,
            num_agents=num_agents,
        )
        agent_state = obs_dict["state"]
        agent_pos = agent_state[..., :2]
        agent_rot = agent_state[..., 2:3]
        agent_vel = agent_state[..., 3:5]
        agent_vel = torch.linalg.vector_norm(agent_vel, dim=-1).unsqueeze(-1)
        agent_ang_vel = agent_state[..., 5:6]
        state = torch.cat((agent_vel, agent_ang_vel), dim=-1)

        goal = obs_dict["goal"]  # x,y
        # NOTE: to_polar's 4th positional arg is `epsilon` (a tiny scalar). This used
        # to pass `state[..., :2]` (the agent velocity) there, which corrupted the
        # goal direction normalization (norm + velocity instead of norm + 1e-8).
        goal_polar = to_polar(goal, agent_pos, agent_rot)

        if "global_path" in obs_dict and self.set_gp_as_goal:
            global_path = obs_dict["global_path"]
            goal_polar = global_path[
                ..., 2:4
            ]  # x,y, dist, angle_to_point_diff, angle_diff_to_next

        # Ablation test: set goal to zero
        # goal_polar = torch.zeros_like(goal_polar)

        state_data = torch.cat((goal_polar, state), dim=-1)
        if "global_path" in obs_dict and self.use_global_path_obs:
            global_path = obs_dict["global_path"][..., 2:]
            state_data = torch.cat((state_data, global_path), dim=-1)

        lidar_dist = obs_dict["lidar_dist"]
        # lidar_vel = obs_dict["lidar_vel"]  # in global, but we want local velocity

        # Build a 2D rotation matrix R(theta) per agent, used to rotate global-frame
        # vectors into each agent's local frame:
        #   R = [[cos, -sin], [sin, cos]]
        # agent_rot has shape (B, 1) = (120, 1) for the W=10, N=12 config. The stack
        # along dim=-1 gives (B, 1, 4), then view(..., 2, 2) lands at (B, 1, 2, 2).
        # Currently only wired up for the commented-out lidar_vel path below; the
        # "other_agents" branch does its own polar conversion via to_polar.
        cos_rot = torch.cos(agent_rot)
        sin_rot = torch.sin(agent_rot)
        rotation_matrix = torch.stack(
            (cos_rot, -sin_rot, sin_rot, cos_rot), dim=-1
        )  # (B, 1, 4) — the middle dim is named "hist" by legacy; with rot shape (B,1) it's 1.
        rotation_matrix = rotation_matrix.view(
            rotation_matrix.shape[0], rotation_matrix.shape[1], 2, 2
        )  # (B, 1, 2, 2)

        # # Reshape lidar_vel to align with the rotation matrices
        # lidar_vel = lidar_vel.view(
        #     lidar_vel.shape[0], lidar_vel.shape[1], lidar_vel.shape[2], 2, 1
        # )  # (batch, hist, rays, 2, 1)
        #
        # # Apply the rotation to each velocity vector
        # lidar_vel_local = torch.matmul(rotation_matrix.unsqueeze(2), lidar_vel).squeeze(
        #     -1
        # )  # (batch, hist, rays, 2)
        other_agents_graph = None
        if "other_agents" in obs_dict:
            # other_agents arrives as (B, N-1, 4): [dx, dy, vx, vy] per peer, ego excluded.
            # For W=10, N=12 that's (120, 11, 4). We are on a flat batch (worlds*agents)
            # here — each ego agent is one row of B.
            other_agents = obs_dict["other_agents"]
            # Convert peer offsets into the ego's local frame, expressed in polar (dist, angle).
            # unsqueeze(-2) / unsqueeze(1) broadcasts ego pos/rot across the peer axis:
            #   agent_pos (B, 2) = (120, 2)      -> (120, 1, 2)   matches (120, 11, 2)
            #   agent_rot (B, 1) = (120, 1)      -> (120, 1, 1) expanded to (120, 11, 1)
            # BUGFIX (M1): the peer block stores each peer's ABSOLUTE world-frame
            # velocity (the old "already relative" comment was false — the scenario
            # packs raw a.state.vel). Meanwhile the peer POSITION is converted to the
            # ego's heading frame (ego-local polar) below. To keep position and
            # velocity in the SAME frame, make the velocity (a) relative to the ego
            # (peer_vel - ego_vel) and (b) rotated by R(-theta) into the ego's heading
            # frame, so the GNN can read approaching-vs-receding from [bearing, v].
            _ego_vel_world = agent_state[..., 3:5]                      # (B, 2) raw 2D
            _rel_vel_world = other_agents[..., 2:] - _ego_vel_world.unsqueeze(-2)
            _cos_e = torch.cos(agent_rot)                               # (B, 1)
            _sin_e = torch.sin(agent_rot)                               # (B, 1)
            _vx = _rel_vel_world[..., 0]                                # (B, N-1)
            _vy = _rel_vel_world[..., 1]
            relative_velocities = torch.stack(
                (_cos_e * _vx + _sin_e * _vy, -_sin_e * _vx + _cos_e * _vy), dim=-1
            )  # (B, N-1, 2) ego-frame relative velocity, consistent with the bearing

            # BUGFIX: to_polar(target, agent_pose, ...) internally computes
            # dist = ||agent_pose - target|| and the bearing of (target - agent_pose),
            # i.e. it expects an ABSOLUTE target position (exactly how the goal is passed
            # at the goal_polar call above). Previously this passed a pre-subtracted
            # (peer - ego) offset, so to_polar subtracted ego A SECOND TIME, yielding
            # dist = ||2*ego - peer|| and a phantom bearing — correct only when ego sits
            # at the origin. other_agents[..., :2] is the ABSOLUTE peer position (the
            # scenario packs absolute a.state.pos), so pass it directly.
            relative_positions_polar = to_polar(
                other_agents[..., :2],
                agent_pos.unsqueeze(1),
                agent_rot.unsqueeze(1).expand(-1, num_agents - 1, -1),
            )
            # as we are tracking the edge of other agents and not center we need to subtract the radius
            relative_positions_polar[..., 0] = relative_positions_polar[
                ..., 0
            ] - self.dynamics.get("agent_radius", 0.2)
            other_agents = torch.cat(
                (relative_positions_polar, relative_velocities), dim=-1
            )  # (B, N-1, 4) = [dist, angle, vx, vy] in ego-local polar+cartesian-vel
            if use_other_agents:
                # ---- Build a sparse PyG graph over "close, non-stationary" neighbors. ----
                # Each ego in the flat batch B becomes one graph. Non-ego peers that pass the
                # proximity+nonzero-vel filter become its nodes. Egos with zero qualifying peers
                # have NO graph entry at all; the caller handles them via the zero-init in
                # forward() and the scatter-back by `global_bat`.
                #
                # Typical numbers for W=10, N=12 (B=120): out of 120*11=1320 possible peer
                # edges, only ~15-25 survive the distance+nonzero-vel filter early in
                # training. That yields ~15 unique egos with at least one neighbor, so
                # global_bat has ~15 entries out of 120. The rest stay zero-filled.
                #
                # Why the two-index dance (graph_batch vs graph_batch_inc):
                #   - graph_batch: mask.nonzero()[:,0] -> global ego index in [0, B)=[0,120).
                #     Sparse; one entry per surviving peer. Used later to scatter GNN pooled
                #     embeddings back into the dense (B, gnn_emb_size) tensor.
                #   - graph_batch_inc: re-indexes the unique egos to a contiguous [0, G) range,
                #     which is what torch_geometric.nn.global_*_pool expects (one int per node
                #     naming which of the G graphs it belongs to). G = len(graph_batch.unique()).
                # So: graph_batch says "which ego in B", graph_batch_inc says "which pooled
                # output slot", and global_bat = graph_batch.unique() closes the loop by
                # telling forward() where to write each pooled embedding back in B.

                # Filter out distant nodes
                distance_threshold = self.dynamics.get("lidar_obs_range")
                mask = other_agents[..., 0] < distance_threshold

                # filter out zero velocities
                mask = mask & (
                    torch.linalg.vector_norm(other_agents[..., 2:], dim=-1) > 0.0
                )
                # Dense, sync-free formulation: keep every peer plus a boolean
                # mask and let the GNN do a masked sum (AgentGraphNet.forward_dense).
                # This is exactly equivalent to the old sparse graph +
                # global_add_pool (sum over each ego's qualifying peers) but avoids
                # the data-dependent torch.argwhere()/unique() that forced a
                # device->host sync on every forward pass.
                other_agents_graph = (other_agents, mask)

        lidar_dist, state_data, other_agents_graph = self.normalize(
            lidar_dist, state_data, other_agents_graph
        )

        return lidar_dist, state_data, other_agents_graph


class LocalNavigationGraphNetDist(LocalNavigationNetBase):
    def __init__(
        self,
        lidar_history_len,
        lidar_num_rays,
        state_input_dim,
        conv_channels,
        kernel_sizes,
        gnn_emb_size,
        output_dim,
        dynamics: typing.Dict,
        use_global_path_obs=False,
        set_gp_as_goal=False,
        std_max=-1,
        std_min=1e-7,
        device="cpu",
        gnn_attention="none",
        **kwargs,
    ):
        super().__init__(
            lidar_history_len,
            lidar_num_rays,
            state_input_dim,
            conv_channels,
            kernel_sizes,
            output_dim,
            dynamics,
            use_global_path_obs,
            set_gp_as_goal,
            device,
            std_max,
            std_min,
        )
        self.gnn_emb_size = gnn_emb_size
        self.lidar_static_encoder = LIDARDistEncoder(
            lidar_num_rays, 1, conv_channels, kernel_sizes, 40
        )  # only current
        self.lidar_dynamic_encoder = LIDARDistEncoder(
            lidar_num_rays, lidar_history_len, conv_channels, kernel_sizes, 20
        )  # the last lidar_history_len frames
        self.agent_gnn = AgentGraphNet(
            4, gnn_emb_size, attention_mode=gnn_attention
        )  # Assuming 4 input features for agents (dist, angle, vel_x, vel_y)

        combined_dim = 60 + state_input_dim + gnn_emb_size
        if use_global_path_obs:
            combined_dim += 3
        self.MLP = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(0.01),
            # nn.Linear(256, 128),
            # nn.LeakyReLU(0.01),
            # nn.Linear(128, output_dim),
        )

        self.scale_net = nn.Linear(256, output_dim)
        self.action_net = nn.Linear(256, output_dim)

        self.obs_to_dict = observation_to_dict
        self.normalize_observation = normalize_observation
        # output dim = 2 * num_actions
        # first num_actions is the mean of the action distribution
        # second num_actions is the std of the action distribution
        self.to(device)

    def forward(self, obs: torch.Tensor):
        # Canonical call shape from torchrl: obs = (worlds, agents, F). During training
        # there is also a "time" axis, e.g. (worlds, time, agents, F), which is why we
        # carry `*batch` and unflatten with the same list at the end.
        #
        # Concrete example (config from LidarSingleStep, W=10, N=12, F=417):
        #     obs.shape              = (10, 12, 417)
        #     *batch, feat           = ([10, 12], 417)
        #     num_agents             = batch[-1] = 12
        #     obs after flatten(0,1) = (120, 417)       # B = 10 * 12 = 120
        #     prepare_obs returns    lidar_dist (120, 3, 120), state_data (120, 7),
        #                            graph with x=(22, 4), batch=(22,), global_bat=(17,)
        #     after MLP/action/scale     (120, 2) for action and scale each
        #     unflatten back to batch    (10, 12, 2) for action, (10, 12, 2) for scale
        #     concatenated output        (10, 12, 4) = [v, w, scale_v, scale_w]
        # With an extra time dim (10, 8, 12, 417) the same trick collapses to B=960.
        #
        # Flatten-then-unflatten pattern (THIS IS THE "BATCH ABUSE"):
        #   1. Peel off the last dim (features). Everything left is "batch stuff".
        #   2. Collapse ALL leading dims into a single axis of size B = prod(batch).
        #      Every downstream op (prepare_obs, lidar conv, GNN scatter) treats B as
        #      the flat ego-agent index.
        #   3. After the MLP, unflatten back to the original leading shape so the output
        #      tensor aligns one-to-one with the input (TorchRL needs this).
        # Keeping the collapse localized to forward() means downstream code can assume a
        # simple 2D (B, F) input regardless of whether the caller is collecting, training,
        # or evaluating.
        *batch, feat = obs.shape
        obs_device = obs.device
        if self.device != obs_device:
            obs = obs.to(self.device)
        num_agents = batch[-1]  # last batch dim is always "agents" by torchrl convention
        if len(batch) > 1:
            obs = obs.flatten(0, len(batch) - 1)  # (..., agents, F) -> (B, F), B=prod(batch)

        lidar_dist, state_data, other_agents_graph = self.prepare_obs(obs, num_agents)
        # lidar_dist: (B, H, R) = (120, 3, 120) in the W=10 N=12 example.
        # Static encoder takes only the most-recent scan (B, 1, R) = (120, 1, 120).
        lidar_static_features = self.lidar_static_encoder(
            lidar_dist[..., -1, None, :]
        )  # only the current lidar data
        lidar_dynamic_features = self.lidar_dynamic_encoder(
            lidar_dist
        )  # the last lidar_history_len frames
        # Peer aggregation: masked sum over each ego's qualifying peers
        # (AgentGraphNet.forward_dense). Egos with no qualifying peers sum to zero,
        # matching the previous zero-filled buffer. None: single-agent / no peers.
        other_agents = torch.zeros(
            lidar_dist.shape[0], self.gnn_emb_size, device=self.device
        )
        if other_agents_graph is not None:
            peers, peer_mask = other_agents_graph
            other_agents = self.agent_gnn.forward_dense(peers, peer_mask)

        combined_features = torch.cat(
            (
                lidar_static_features,
                lidar_dynamic_features,
                state_data,
                other_agents,
            ),
            dim=-1,
        )
        emb = self.MLP(combined_features)
        action = self.action_net(emb)
        scale = self.scale_net(emb)
        if len(batch) > 1:
            # Mirror of the flatten in step 2 above:
            #   (B=120, 2) -> (10, 12, 2) with batch=[10, 12]
            # or (B=960, 2) -> (10, 8, 12, 2) with batch=[10, 8, 12].
            # Must stay symmetric with that flatten() or torchrl's tensordict will error.
            action = action.unflatten(0, batch)
            scale = scale.unflatten(0, batch)

        if self.is_critic:
            v = action.to(obs_device)
            return v  # return the value function

        linear_action = F.sigmoid(
            action[..., None, 0]
        )  # linear action is between 0 and 1
        angular_action = F.tanh(
            action[..., None, 1]
        )  # angular action is between -1 and 1
        scale = F.softplus(scale).clamp_min(self.std_min)  # scale is always positive
        if self.std_max != -1:
            scale = scale.clamp_max(self.std_max)  # ceiling: stops entropy bonus from
            #                                        inflating std unboundedly (it has no
            #                                        ceiling otherwise -> std explodes)
        output = torch.cat((linear_action, angular_action, scale), dim=-1)
        output = output.to(obs_device)
        return output


class LocalNavigationNetDist(LocalNavigationNetBase):
    def __init__(
        self,
        lidar_history_len,
        lidar_num_rays,
        state_input_dim,
        conv_channels,
        kernel_sizes,
        output_dim,
        dynamics: typing.Dict,
        use_global_path_obs=False,
        set_gp_as_goal=False,
        std_max=-1,
        std_min=1e-7,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            lidar_history_len,
            lidar_num_rays,
            state_input_dim,
            conv_channels,
            kernel_sizes,
            output_dim,
            dynamics,
            use_global_path_obs,
            set_gp_as_goal,
            device,
            std_max,
        )

        self.lidar_static_encoder = LIDARDistEncoder(
            lidar_num_rays, 1, conv_channels, kernel_sizes, 40
        )  # only current
        self.lidar_dynamic_encoder = LIDARDistEncoder(
            lidar_num_rays, lidar_history_len, conv_channels, kernel_sizes, 20
        )  # the last lidar_history_len frames
        combined_dim = 60 + state_input_dim
        if use_global_path_obs:
            combined_dim += 3
        self.MLP = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, output_dim),
        )
        self.obs_to_dict = observation_to_dict
        self.normalize_observation = normalize_observation
        # output dim = 2 * num_actions
        # first num_actions is the mean of the action distribution
        # second num_actions is the std of the action distribution
        self.to(device)

    def forward(self, obs: torch.Tensor):
        *batch, feat = obs.shape
        obs_device = obs.device
        if self.device != obs_device:
            obs = obs.to(self.device)
        num_agents = batch[-1]
        if len(batch) > 1:
            obs = obs.flatten(0, len(batch) - 1)
        lidar_dist, state_data, _ = self.prepare_obs(obs, num_agents)
        lidar_static_features = self.lidar_static_encoder(
            lidar_dist[..., -1, None, :]
        )  # only the current lidar data
        lidar_dynamic_features = self.lidar_dynamic_encoder(
            lidar_dist
        )  # the last lidar_history_len frames
        combined_features = torch.cat(
            (lidar_static_features, lidar_dynamic_features, state_data), dim=-1
        )
        output = self.MLP(combined_features)

        if len(batch) > 1:
            output = output.unflatten(0, batch)

        if self.is_critic:
            output = output.to(obs_device)
            return output  # return the value function

        linear_action = F.sigmoid(
            output[..., None, 0]
        )  # linear action is between 0 and 1
        angular_action = F.tanh(
            output[..., None, 1]
        )  # angular action is between -1 and 1
        scale = F.softplus(
            output[..., output.shape[-1] // 2 :]
        )  # scale is always positive
        if self.std_max != -1:
            scale = torch.clamp(scale, min=self.std_min, max=self.std_max)
        output = torch.cat((linear_action, angular_action, scale), dim=-1)
        output = output.to(obs_device)
        return output


class LocalNavigationNetDistV2(LocalNavigationNetBase):
    def __init__(
        self,
        lidar_history_len,
        lidar_num_rays,
        state_input_dim,
        conv_channels,
        kernel_sizes,
        output_dim,
        dynamics: typing.Dict,
        use_global_path_obs=False,
        set_gp_as_goal=False,
        std_max=-1,
        std_min=1e-7,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            lidar_history_len,
            lidar_num_rays,
            state_input_dim,
            conv_channels,
            kernel_sizes,
            output_dim,
            dynamics,
            use_global_path_obs,
            set_gp_as_goal,
            device,
            std_max,
            std_min,
        )

        self.lidar_static_encoder = LIDARDistEncoder(
            lidar_num_rays, 1, conv_channels, kernel_sizes, 40
        )  # only current
        self.lidar_dynamic_encoder = LIDARDistEncoder(
            lidar_num_rays, lidar_history_len, conv_channels, kernel_sizes, 20
        )  # the last lidar_history_len frames
        combined_dim = 60 + state_input_dim
        if use_global_path_obs:
            combined_dim += 3
        self.MLP = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(0.01),
        )

        self.scale_net = nn.Linear(256, output_dim)
        self.action_net = nn.Linear(256, output_dim)

        self.obs_to_dict = observation_to_dict
        self.normalize_observation = normalize_observation
        # output dim = 2 * num_actions
        # first num_actions is the mean of the action distribution
        # second num_actions is the std of the action distribution
        self.to(device)

    def forward(self, obs: torch.Tensor):
        *batch, feat = obs.shape
        obs_device = obs.device
        if self.device != obs_device:
            obs = obs.to(self.device)
        num_agents = batch[-1]
        if len(batch) > 1:
            obs = obs.flatten(0, len(batch) - 1)

        lidar_dist, state_data, _ = self.prepare_obs(
            obs, num_agents, use_other_agents=False
        )
        lidar_static_features = self.lidar_static_encoder(
            lidar_dist[..., -1, None, :]
        )  # only the current lidar data
        lidar_dynamic_features = self.lidar_dynamic_encoder(
            lidar_dist
        )  # the last lidar_history_len frames

        combined_features = torch.cat(
            (lidar_static_features, lidar_dynamic_features, state_data),
            dim=-1,
        )
        emb = self.MLP(combined_features)
        action = self.action_net(emb)
        scale = self.scale_net(emb)
        if len(batch) > 1:
            action = action.unflatten(0, batch)
            scale = scale.unflatten(0, batch)

        if self.is_critic:
            v = action.to(obs_device)
            return v  # return the value function

        linear_action = F.sigmoid(
            action[..., None, 0]
        )  # linear action is between 0 and 1
        angular_action = F.tanh(
            action[..., None, 1]
        )  # angular action is between -1 and 1
        scale = F.softplus(scale).clamp_min(self.std_min)  # scale is always positive
        if self.std_max != -1:
            scale = scale.clamp_max(self.std_max)  # ceiling: stops entropy bonus from
            #                                        inflating std unboundedly (it has no
            #                                        ceiling otherwise -> std explodes)
        output = torch.cat((linear_action, angular_action, scale), dim=-1)
        output = output.to(obs_device)
        return output


class MultiAgentNetBase(nn.Module):
    """A base class for multi-agent networks."""

    _empty_net: nn.Module

    def __init__(
        self,
        *,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        agent_dim: int,
        vmap_randomness: str = "different",
        **kwargs,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.share_params = share_params
        self.centralised = centralised
        self.agent_dim = agent_dim
        self._vmap_randomness = vmap_randomness

        agent_networks = [
            self._build_single_net(**kwargs)
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        initialized = True
        for p in agent_networks[0].parameters():
            if isinstance(p, torch.nn.UninitializedParameter):
                initialized = False
                break
        self.initialized = initialized
        self._make_params(agent_networks)
        kwargs["device"] = "meta"
        self.__dict__["_empty_net"] = self._build_single_net(**kwargs)

    @property
    def vmap_randomness(self):
        if self.initialized:
            return self._vmap_randomness
        # The class _BatchedUninitializedParameter and buffer are not batched
        # by vmap so using "different" will raise an exception because vmap can't find
        # the batch dimension. This is ok though since we won't have the same config
        # for every element (as one might expect from "same").
        return "same"

    def _make_params(self, agent_networks):
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)

    @abc.abstractmethod
    def _build_single_net(self, *, device, **kwargs): ...

    @abc.abstractmethod
    def _pre_forward_check(self, inputs): ...

    @staticmethod
    def vmap_func_module(module, *args, **kwargs):
        def exec_module(params, *input):
            with params.to_module(module):
                return module(*input)

        return torch.vmap(exec_module, *args, **kwargs)

    def forward(self, *inputs: typing.Tuple[torch.Tensor]) -> torch.Tensor:
        # Multi-agent dispatch — there are four shape regimes depending on
        # (share_params, centralised):
        #   share=True,  centralised=False: ONE net applied in parallel to all agents.
        #       The single-agent net is itself expected to handle an extra "agents" axis
        #       (our LocalNavigation* nets do exactly that via the flatten/unflatten trick
        #       in their own forward). This is the fast path in the main training loop.
        #   share=True,  centralised=True: one net, one shared output, broadcast to every
        #       agent (useful for the critic in centralised training).
        #   share=False, centralised=False: n_agents independent nets, vmapped over the
        #       agents axis (self.agent_dim = -2). Each agent sees only its own obs slice.
        #   share=False, centralised=True: n_agents independent nets, each seeing the full
        #       concatenated obs (in_dims (0, None) means "don't vmap the inputs").
        # The shape (..., agents, F) in / (..., agents, out) out contract is preserved in
        # all four regimes; the divergent logic lives in how the nets are applied.
        if len(inputs) > 1:
            inputs = torch.cat([*inputs], -1)
        else:
            inputs = inputs[0]

        inputs = self._pre_forward_check(inputs)
        # If parameters are not shared, each agent has its own network
        if not self.share_params:
            if self.centralised:
                output = self.vmap_func_module(
                    self._empty_net, (0, None), (-2,), randomness=self.vmap_randomness
                )(self.params, inputs)
            else:
                output = self.vmap_func_module(
                    self._empty_net,
                    (0, self.agent_dim),
                    (-2,),
                    randomness=self.vmap_randomness,
                )(self.params, inputs)

            if output.shape[-2] != (self.n_agents):
                raise ValueError(
                    f"Multi-agent network expected output with shape[-2]={self.n_agents}"
                    f" but got {output.shape}"
                )
        # If parameters are shared, agents use the same network
        else:
            with self.params.to_module(self._empty_net):
                output = self._empty_net(inputs)

            if self.centralised:
                # If the parameters are shared, and it is centralised, all agents will have the same output
                # We expand it to maintain the agent dimension, but values will be the same for all agents
                n_agent_outputs = output.shape[-1]
                output = output.view(*output.shape[:-1], n_agent_outputs)
                output = output.unsqueeze(-2)
                output = output.expand(
                    *output.shape[:-2], self.n_agents, n_agent_outputs
                )

        return output

    def reset_parameters(self):
        """Resets the parameters of the model."""

        def vmap_reset_module(module, *args, **kwargs):
            def reset_module(params):
                with params.to_module(module):
                    _reset_parameters_recursive(module)
                    return params

            return torch.vmap(reset_module, *args, **kwargs)

        if not self.share_params:
            vmap_reset_module(self._empty_net, randomness="different")(self.params)
        else:
            with self.params.to_module(self._empty_net):
                _reset_parameters_recursive(self._empty_net)


class MultiAgentLocalNavNet(MultiAgentNetBase):
    def __init__(
        self,
        base_net: str,
        lidar_history_len,
        lidar_input_dim,
        state_input_dim,
        conv_channels,
        kernel_sizes,
        n_agent_outputs,
        n_agents: int,
        share_params: bool,
        use_global_path_obs: bool,
        set_gp_as_goal: bool,
        dynamics: typing.Dict,
        device: typing.Optional[DEVICE_TYPING] = None,
        std_max=-1,
        **kwargs,
    ):
        self.base_net = base_net
        self.n_agents = n_agents
        self.share_params = share_params
        self.device = device
        self.agent_dim = -2
        self.n_agent_outputs = n_agent_outputs
        self.lidar_history_len = lidar_history_len
        self.lidar_input_dim = lidar_input_dim
        self.state_input_dim = state_input_dim
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dynamics = dynamics
        self.use_global_path_obs = use_global_path_obs
        self.set_gp_as_goal = set_gp_as_goal
        self.std_max = std_max
        self.std_min = kwargs.get("std_min", 1e-7)
        self.gnn_emb_size = kwargs.get("gnn_emb_size", None)
        self.gnn_attention = kwargs.get("gnn_attention", "none")

        # all the self. variables are used in the _build_single_net method so they need to be defined before calling it
        super(MultiAgentLocalNavNet, self).__init__(
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            agent_dim=-2,
            **kwargs,
        )

    def _build_single_net(self, *, device, **kwargs):
        if self.base_net == "oursD":
            return LocalNavigationNetDist(
                self.lidar_history_len,
                self.lidar_input_dim,
                self.state_input_dim,
                self.conv_channels,
                self.kernel_sizes,
                self.n_agent_outputs,
                dynamics=self.dynamics,
                use_global_path_obs=self.use_global_path_obs,
                set_gp_as_goal=self.set_gp_as_goal,
                std_max=self.std_max,
                std_min=self.std_min,
                device=self.device,
            )
        elif self.base_net == "oursGraph":
            assert self.gnn_emb_size is not None, "GNN embedding size must be provided"
            return LocalNavigationGraphNetDist(
                self.lidar_history_len,
                self.lidar_input_dim,
                self.state_input_dim,
                self.conv_channels,
                self.kernel_sizes,
                self.gnn_emb_size,
                self.n_agent_outputs,
                dynamics=self.dynamics,
                use_global_path_obs=self.use_global_path_obs,
                set_gp_as_goal=self.set_gp_as_goal,
                std_max=self.std_max,
                std_min=self.std_min,
                device=self.device,
                gnn_attention=self.gnn_attention,
            )
        else:
            raise ValueError(f"Unknown base net: {self.base_net}")

    def set_std_max(self, std_max):
        self.std_max = std_max
        # BUGFIX: the share_params forward executes through self._empty_net, and the
        # action-distribution std ceiling is clamped using THAT inner net's `std_max`
        # attribute. `std_max` is a plain (non-parameter) attribute, so params.to_module()
        # never syncs it — updating only the wrapper here made STDDecay annealing a silent
        # no-op (the ceiling stayed at the build-time value forever). Propagate it.
        _en = self.__dict__.get("_empty_net", None)
        if _en is not None and hasattr(_en, "std_max"):
            _en.std_max = std_max
        for _net in (self.__dict__.get("_agent_networks", None) or []):
            if hasattr(_net, "std_max"):
                _net.std_max = std_max
        # self.params.std_max = nn.Parameter(torch.tensor(std_max, device=self.device), requires_grad=False)

    def _pre_forward_check(self, inputs: torch.Tensor):
        # If the model is centralized, agents have full observability
        if self.centralised:
            inputs = inputs.flatten(-2, -1)
        # if inputs.device != self.device:
        #     inputs = inputs.to(self.device)
        return inputs
