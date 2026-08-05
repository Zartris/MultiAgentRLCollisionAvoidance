import math
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from scenario.CollisionAvoidance_base import normalize_observation, observation_to_dict


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = (
            -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std
    )  # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True)  # num_env * frames * 1
    return log_density


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 1, -1)


def get_laser_observation(scan: torch.Tensor):
    # scan is inverted, so we invert it back
    scan = 6.0 - scan
    # all readings over 3.6 are set to 6.0
    scan = torch.where(scan > 6.0, 6.0, scan)
    return scan / 6.0 - 0.5


def get_local_goal(state, goal_point):
    x = state[..., 0]
    y = state[..., 1]
    theta = state[..., 2]
    goal_x = goal_point[..., 0]
    goal_y = goal_point[..., 1]
    local_x = (goal_x - x) * torch.cos(theta) + (goal_y - y) * torch.sin(theta)
    local_y = -(goal_x - x) * torch.sin(theta) + (goal_y - y) * torch.cos(theta)
    return torch.stack([local_x, local_y], dim=-1)


class CNNPolicy(nn.Module):
    def __init__(
            self,
            lidar_history_len,
            lidar_input_dim,
            set_gp_as_goal,
            dynamics,
            is_critic,
            device,
    ):
        super(CNNPolicy, self).__init__()
        assert lidar_input_dim == 512, "Only support 512 lidar input"
        self.lidar_input_dim = lidar_input_dim
        self.lidar_history_len = lidar_history_len
        self.set_gp_as_goal = set_gp_as_goal
        self.dynamics = dynamics
        self.is_critic = is_critic
        self.device = device

        self.obs_to_dict = observation_to_dict

        self.logstd = nn.Parameter(torch.zeros(2))

        self.lidar_conv1 = nn.Conv1d(
            in_channels=lidar_history_len,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=1,
        )
        self.lidar_conv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.fc1 = nn.Linear(128 * 32, 256)
        self.fc2 = nn.Linear(256 + 2 + 2, 128)
        self.out1 = nn.Linear(128, out_features=1)
        if not is_critic:
            self.out2 = nn.Linear(128, 1)

        self.to(device)
        self.device = device

    def prepare_obs(self, obs, num_agents):
        obs_dict = self.obs_to_dict(
            obs,
            True,
            self.lidar_history_len,
            self.lidar_input_dim,
            use_global_path=self.set_gp_as_goal,
            num_agents=num_agents,
        )
        agent_state = obs_dict["state"]  # x, y, theta, linear_speed, angular_speed
        state = agent_state[..., 3:]  # vel_x, vel_y, angular_speed
        linear = torch.linalg.vector_norm(state[..., :2], dim=-1).unsqueeze(-1)
        angular = state[..., 2:3]
        state = torch.cat((linear, angular), dim=-1)
        goal = obs_dict["goal"]  # x,y
        if "global_path" in obs_dict and self.set_gp_as_goal:
            global_path = obs_dict["global_path"]
            goal = global_path[..., :2]
            goal = get_local_goal(agent_state[..., :3], goal)
        else:
            goal = get_local_goal(agent_state[..., :3], goal)
        lidar_data = get_laser_observation(obs_dict["lidar_dist"])
        return lidar_data, goal, state

    def forward(self, obs):
        """
        returns value estimation, action, log_action_prob
        input:
            lidar_data: (batch, lidar_history_len, lidar_input_dim)
            state_data: (batch,  linear_speed, angular_speed)
            goal: (batch, goal_x_angle, goal_y_angle)
        """
        *batch, feat = obs.shape
        obs_device = obs.device
        if self.device != obs_device:
            obs = obs.to(self.device)
        num_agents = batch[-1]
        if len(batch) > 1:
            obs = obs.flatten(0, len(batch) - 1)
        lidar_data, goal, state = self.prepare_obs(obs, num_agents)
        # action
        a = F.relu(self.lidar_conv1(lidar_data))
        a = F.relu(self.lidar_conv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.fc1(a))

        a = torch.cat((a, goal, state), dim=-1)
        a = F.relu(self.fc2(a))
        if self.is_critic:
            v = self.out1(a)
            if len(batch) > 1:
                v = v.unflatten(0, batch)
            return v

        linear_action = F.sigmoid(self.out1(a))
        angular_action = F.tanh(self.out2(a))
        if len(batch) > 1:
            linear_action = linear_action.unflatten(0, batch)
            angular_action = angular_action.unflatten(0, batch)
        action = torch.cat((linear_action, angular_action), dim=-1)
        logstd = self.logstd.expand_as(action)
        std = torch.exp(logstd)
        output = torch.cat((linear_action, angular_action, std), dim=-1)
        output = output.to(obs_device)
        return output


class CNNPolicyCombined(nn.Module):
    def __init__(
            self, lidar_history_len, lidar_input_dim, set_gp_as_goal, dynamics, is_critic
    ):
        super(CNNPolicyCombined, self).__init__()
        assert lidar_input_dim == 512, "Only support 512 lidar input"
        self.lidar_input_dim = lidar_input_dim
        self.lidar_history_len = lidar_history_len
        self.set_gp_as_goal = set_gp_as_goal
        self.dynamics = dynamics
        self.is_critic = is_critic

        self.obs_to_dict = observation_to_dict

        self.logstd = nn.Parameter(torch.zeros(2))

        self.act_fea_cv1 = nn.Conv1d(
            in_channels=lidar_history_len,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=1,
        )
        self.act_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.act_fc1 = nn.Linear(128 * 32, 256)
        self.act_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.actor1 = nn.Linear(128, out_features=1)
        self.actor2 = nn.Linear(128, 1)

        self.crt_fea_cv1 = nn.Conv1d(
            in_channels=lidar_history_len,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=1,
        )
        self.crt_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.crt_fc1 = nn.Linear(128 * 32, 256)
        self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.critic = nn.Linear(128, 1)

    def prepare_obs(self, obs, num_agents):
        obs_dict = self.obs_to_dict(
            obs,
            True,
            self.lidar_history_len,
            self.lidar_input_dim,
            use_global_path=self.set_gp_as_goal,
            num_agents=num_agents,
        )
        agent_state = obs_dict["state"]  # x, y, theta, linear_speed, angular_speed
        state = agent_state[..., 3:]  # vel_x, vel_y, angular_speed
        linear = torch.linalg.vector_norm(state[..., :2], dim=-1).unsqueeze(-1)
        angular = state[..., 2:3]
        state = torch.cat((linear, angular), dim=-1)
        goal = obs_dict["goal"]  # x,y
        if "global_path" in obs_dict and self.set_gp_as_goal:
            global_path = obs_dict["global_path"]
            goal = global_path[..., :2]
            goal = get_local_goal(agent_state[..., :3], goal)
        else:
            goal = get_local_goal(agent_state[..., :3], goal)
        lidar_data = get_laser_observation(obs_dict["lidar"])
        return lidar_data, goal, state

    def forward(self, obs):
        """
        returns value estimation, action, log_action_prob
        input:
            lidar_data: (batch, lidar_history_len, lidar_input_dim)
            state_data: (batch,  linear_speed, angular_speed)
            goal: (batch, goal_x_angle, goal_y_angle)
        """
        *batch, feat = obs.shape
        num_agents = batch[-1]
        if len(batch) > 1:
            obs = obs.flatten(0, len(batch) - 1)
        lidar_data, goal, state = self.prepare_obs(obs, num_agents)
        # action
        a = F.relu(self.act_fea_cv1(lidar_data))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, state), dim=-1)
        a = F.relu(self.act_fc2(a))
        linear_action = F.sigmoid(self.actor1(a))  # linear action
        angular_action = F.tanh(self.actor2(a))  # angular action
        if len(batch) > 1:
            linear_action = linear_action.unflatten(0, batch)
            angular_action = angular_action.unflatten(0, batch)
        # mean = torch.cat((mean1, mean2), dim=-1)
        action = torch.cat((linear_action, angular_action), dim=-1)
        logstd = self.logstd.expand_as(action)
        std = torch.exp(logstd)
        # action = torch.normal(mean, std)

        # action prob on log scale
        # logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        # v = F.relu(self.crt_fea_cv1(x))
        # v = F.relu(self.crt_fea_cv2(v))
        # v = v.view(v.shape[0], -1)
        # v = F.relu(self.crt_fc1(v))
        # v = torch.cat((v, goal, speed), dim=-1)
        # v = F.relu(self.crt_fc2(v))
        # v = self.critic(v)
        out = torch.cat((linear_action, angular_action, std), dim=-1)
        return out

    # def evaluate_actions(self, x, goal, speed, action):
    #     v, _, _, mean = self.forward(x, goal, speed)
    #     logstd = self.logstd.expand_as(mean)
    #     std = torch.exp(logstd)
    #     # evaluate
    #     logprob = log_normal_density(action, mean, log_std=logstd, std=std)
    #     dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
    #     dist_entropy = dist_entropy.sum(-1).mean()
    #     return v, logprob, dist_entropy
