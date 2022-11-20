import random
import os
import numpy as np
from numpy import savetxt, loadtxt
import rewardmodel
from rewardmodel import calculate_reward


def action_select(obs):
    action_space = [1, 2, 3, 4, 5, 6, 7]
    if os.path.exists('theta.csv'):
        weights = loadtxt('theta.csv')
        next_rew_matrix = []
        feature_matrix = []
        feature_matrix = [obs + obs] * len(action_space)
        feature_matrix = np.array(feature_matrix)
        dv = 0.5
        for action in action_space:
            dx = 0
            dy = 0
            dz = 0
            if action == 2:
                dy = 1 * dv  # Left
            if action == 6:
                dy = -1 * dv  # Right
            if action == 1:
                dz = 1 * dv  # Up
            if action == 7:
                dz = -1 * dv  # Down
            if action == 3:
                dx = 1 * dv  # Forward
            if action == 5:
                dx = -1 * dv  # Back
            d = [dx, dy, dz]
            for i in range(3):
                feature_matrix[action-1][i] += d[i]
                feature_matrix[action-1][i+3] += d[i]

        for n in range(len(feature_matrix)):
            next_rew = weights[0]
            for m in range(len(feature_matrix[0])):
                next_rew += feature_matrix[n][m] * weights[m+1]
            next_rew_matrix.append(next_rew)
        print(next_rew_matrix)
        next_action = action_space[next_rew_matrix.index(max(next_rew_matrix))]
        action = next_action
    else:
        action = random.choice(action_space)
    return action
