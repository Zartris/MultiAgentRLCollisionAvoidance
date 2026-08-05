#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.utils import TorchUtils


class MyDiffDrive(Dynamics):
    def __init__(
            self,
            world: vmas.simulator.core.World,
            integration: str = "rk4",  # one of "euler", "rk4"
    ):
        super().__init__()
        assert integration == "rk4" or integration == "euler"

        self.dt = world.dt
        self.integration = integration
        self.world = world

    @property
    def drag(self):
        return self.agent.drag if self.agent.drag is not None else self.world._drag

    def f(self, state, u_command, ang_vel_command):
        theta = state[:, 2]
        dx = u_command * torch.cos(theta)
        dy = u_command * torch.sin(theta)
        dtheta = ang_vel_command
        return torch.stack((dx, dy, dtheta), dim=-1)  # [batch_size,3]

    def euler(self, state, u_command, ang_vel_command):
        return self.dt * self.f(state, u_command, ang_vel_command)

    def runge_kutta(self, state, u_command, ang_vel_command):
        k1 = self.f(state, u_command, ang_vel_command)
        k2 = self.f(state + self.dt * k1 / 2, u_command, ang_vel_command)
        k3 = self.f(state + self.dt * k2 / 2, u_command, ang_vel_command)
        k4 = self.f(state + self.dt * k3, u_command, ang_vel_command)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    def _limit_force(self, force):
        if self.agent.movable:
            if self.agent.max_f is not None:
                force = TorchUtils.clamp_with_norm(force, self.agent.max_f)
            if self.agent.f_range is not None:
                force = torch.clamp(force, -self.agent.f_range, self.agent.f_range)
            return force

    def _limit_torque(self, torque):
        if self.agent.rotatable:
            if self.agent.max_t is not None:
                torque = TorchUtils.clamp_with_norm(torque, self.agent.max_t)
            if self.agent.t_range is not None:
                torque = torch.clamp(torque, -self.agent.t_range, self.agent.t_range)

            return torque

    def process_action(self):
        u_command = self.agent.action.u[:, 0]
        # u_command = torch.where(u_command < 1e-4, 0, u_command)

        ang_vel_command = self.agent.action.u[:, 1]  # Angular velocity

        u_command[self.agent.is_terminated] = 0
        ang_vel_command[self.agent.is_terminated] = 0

        # make sure the combined action is within the limits (left wheel, right wheel velocity)
        v_left = u_command - ang_vel_command * self.agent.shape.radius
        v_right = u_command + ang_vel_command * self.agent.shape.radius

        # make sure the left and right wheel actions are within the limits
        # Check if wheel velocities exceed the maximum limit
        max_vel = torch.max(torch.abs(v_left), torch.abs(v_right))

        # If the max velocity exceeds the limit, scale down the forward velocity
        scaling_factor = torch.where(max_vel > self.agent.v_range, self.agent.v_range / max_vel,
                                     torch.ones_like(max_vel))
        u_command *= scaling_factor

        v_cur_x = self.agent.state.vel[:, 0] * (
                1 - self.drag
        )  # Current velocity in x-direction
        v_cur_y = self.agent.state.vel[:, 1] * (
                1 - self.drag
        )  # Current velocity in y-direction
        v_cur_angular = self.agent.state.ang_vel[:, 0] * (
                1 - self.drag
        )  # Current angular velocity
        cur_vel = torch.linalg.vector_norm(self.agent.state.vel, dim=-1) * (
                1 - self.drag
        )

        u_acceleration = (u_command - cur_vel) / self.dt
        ang_acceleration = (ang_vel_command - v_cur_angular) / self.dt

        u_force = self.agent.mass * u_acceleration
        ang_torque = self.agent.moment_of_inertia * ang_acceleration

        # executable force and torque
        e_force = self._limit_force(u_force)
        e_ang_torque = self._limit_torque(ang_torque)

        e_vel = (e_force / self.agent.mass) * self.dt + cur_vel
        e_ang_vel = (
                            e_ang_torque / self.agent.moment_of_inertia
                    ) * self.dt + v_cur_angular

        # Current state of the agent
        state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)
        # Select the integration method to calculate the change in state
        if self.integration == "euler":
            delta_state = self.euler(state, e_vel, e_ang_vel)
        else:
            delta_state = self.runge_kutta(state, e_vel, e_ang_vel)

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = (delta_state[:, 0] - v_cur_x * self.dt) / self.dt ** 2
        acceleration_y = (delta_state[:, 1] - v_cur_y * self.dt) / self.dt ** 2
        acceleration_angular = (
                                       delta_state[:, 2] - v_cur_angular * self.dt
                               ) / self.dt ** 2

        # Calculate the forces required for the linear accelerations
        force_x = self.agent.mass * acceleration_x
        force_y = self.agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque = self.agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)
