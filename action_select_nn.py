import os
import random
import numpy as np
from reward-functions.rewardmodel_classification import predict


def nn_action_select(obs):
    action_space = [0, 1, 2, 3, 4]
    if os.path.exists("artifacts/features.csv"):
        next_rew_matrix = []
        obs = list(obs)
        feature_matrix = [obs + [0]] * len(action_space)
        feature_matrix = np.array(feature_matrix)
        dv = 0.5
        for action in range(len(action_space)):
            dx = 0
            dy = 0
            dz = 0
            if action == 1:
                dy = 1 * dv  # Left
            if action == 3:
                dy = -1 * dv  # Right
            if action == 2:
                dz = -1 * dv  # Down
            if action == 4:
                dx = 1 * dv  # Forward
            if action == 0:
                dx = -1 * dv  # Back
            d = [dx, dy, dz]
            for i in range(3):
                feature_matrix[action][i] += d[i]
                # feature_matrix[action][i+3] += d[i]
                feature_matrix[action][-1] = action

        for n in range(len(feature_matrix)):
            next_rew_matrix.append(predict(feature_matrix[n]))
        next_action = action_space[next_rew_matrix.index(max(next_rew_matrix))]
    else:
        next_action = random.choice(action_space)
    return next_action
