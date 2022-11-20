import random
import os
import numpy as np
from numpy import savetxt, loadtxt
from knn_reward import find_neighbours


def knn_action_select(obs):
    action_space = [1, 2, 3, 5, 6, 7]
    array_act_space = np.array(action_space)
    next_rew_matrix = [0, 0, 0, 0, 0, 0]
    feature_matrix = []
    feature_matrix = [obs] * len(action_space)
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
            feature_matrix[action_space.index(action)][i] += d[i]
            feature_matrix[action_space.index(action)][i+3] += d[i]
            feature_matrix[action_space.index(action)] = feature_matrix[action_space.index(action)] + [action]

        for n in range(len(feature_matrix)):
            next_rew = find_neighbours(feature_matrix[n], 5)
            next_rew_matrix[n] = next_rew
        idx = next_rew_matrix.index(max(next_rew_matrix))
        next_action = array_act_space[idx]
        final_action = next_action
    return final_action
