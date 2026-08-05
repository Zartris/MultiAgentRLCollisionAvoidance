import os
import profile
import time

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json

import util
from agent import Agent
from network import NetworkVP_rnn, Config, Actions

import os
import time
import zmq
import json
import numpy as np
from agent import Agent
from network import NetworkVP_rnn, Config, Actions

# Initialize ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REP)  # REP for reply mode
socket.bind("tcp://127.0.0.1:5555")  # Bind to a TCP port

# Load your TensorFlow model
possible_actions = Actions()
num_actions = possible_actions.num_actions
nn = NetworkVP_rnn(Config.DEVICE, 'network', num_actions)
checkpoints = os.path.dirname(os.path.realpath(__file__)) + '\\..\\..\\checkpoints\\GA3C_CADRL\\network_01900000'
nn.simple_load(checkpoints)

while True:
    # Wait for the next request from the client
    message = socket.recv()
    data = json.loads(message.decode('utf-8'))

    # Start timing for performance measurement
    t0 = time.perf_counter()

    # Recreate the agent state from the received data
    this_agent = data["agent_state"]
    other_agents_list = data["other_agents"]

    start_x = this_agent[0]
    start_y = this_agent[1]
    goal_x = this_agent[2]
    goal_y = this_agent[3]
    pref_speed = this_agent[4]
    radius = this_agent[5]
    heading_angle = this_agent[6]
    v_x = this_agent[7]
    v_y = this_agent[8]

    host_agent = Agent(start_x, start_y, goal_x, goal_y,
                       radius=radius, pref_speed=pref_speed, initial_heading=heading_angle,
                       id=0)
    host_agent.vel_global_frame = np.array([v_x, v_y])

    other_agents = []
    for i, oa in enumerate(other_agents_list):
        x = oa[0]
        y = oa[1]
        v_x = oa[3]
        v_y = oa[4]
        radius = oa[2]

        other_agent = Agent(x, y, 0, 0, radius=radius, id=i + 1)
        other_agent.vel_global_frame = np.array([v_x, v_y])
        other_agents.append(other_agent)

    obs = host_agent.observe(other_agents)[1:]
    obs = np.expand_dims(obs, axis=0)

    predictions = nn.predict_p(obs)[0]
    raw_action = possible_actions.actions[np.argmax(predictions)]
    # model does not take into consideration rotation speed, so we have to calculate in the dt
    angular_vel = raw_action[1] / 0.1 # 0.1 is the dt
    action = np.array([host_agent.pref_speed * raw_action[0], angular_vel])

    # but we have max rotation speed, so we scale the action
    # Define bounds
    lower_bound = -1
    upper_bound = 1

    # Find the maximum absolute value of the action components
    max_abs_value = np.max(np.abs(action))

    # Calculate the scaling factor
    if max_abs_value > upper_bound:
        scale_factor = upper_bound / max_abs_value
    else:
        scale_factor = 1  # No scaling needed if within bounds

    # Scale the action
    scale_action = action * scale_factor


    print("=========================")
    print("action:", action)
    print("Scaled Action:", scale_action)
    print("Total time:", time.perf_counter() - t0)
    print("=========================")
    # Send the prediction back to the client
    socket.send_json({'predictions': action.tolist()})