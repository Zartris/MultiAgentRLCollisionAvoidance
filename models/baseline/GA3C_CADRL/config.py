import numpy as np


class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT = True
    USE_DROPOUT = False
    USE_REGULARIZATION = True
    ROBOT_MODE = True
    EVALUATE_MODE = True

    SENSING_HORIZON = 8.0

    MIN_POLICY = 1e-4

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 20
    MULTI_AGENT_ARCH = 'RNN'

    DEVICE = '/cpu:0'  # Device

    HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1  # num other agents
    AGENT_ID_LENGTH = 1  # id
    IS_ON_LENGTH = 1  # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5])  # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0])  # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
                                       1.0])  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0,
                                       1.0])  # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR, HOST_AGENT_AVG_VECTOR,
                                             np.tile(OTHER_AGENT_AVG_VECTOR, MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR, HOST_AGENT_STD_VECTOR,
                                             np.tile(OTHER_AGENT_STD_VECTOR, MAX_NUM_OTHER_AGENTS_OBSERVED)])

    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH
