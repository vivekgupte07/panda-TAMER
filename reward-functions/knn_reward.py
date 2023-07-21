import os .path
from math import sqrt
import numpy as np
from numpy import loadtxt, savetxt


#  function to find euclidean distance
def euclidean_dist(features, obs):
    dist = list()
    for i in range(len(features)):
        dist_sq = 0
        for j in range(len(obs)):
            dist_sq += (features[i][j] - obs[j]) ** 2
        dist.append([sqrt(dist_sq), i])
    return dist


# find nearest neighbours
def find_neighbours(obs, k):
    if os.path.exists('features.csv'):
        features = loadtxt('features.csv')
    else:
        features = [[0, 0, 0, 0, 0, 0] * k]

    if os.path.exists('rewards.csv'):
        rew = loadtxt('rewards.csv')
    else:
        rew = [0, 0, 0, 0, 0, 0]

    dist = euclidean_dist(features, obs)
    nearest_distances = sorted(dist)
    neighbours_rews = list()
    for i in range(k):
        neighbours_rews.append(rew[nearest_distances[i][1]])

    reward = max(set(neighbours_rews), key=neighbours_rews.count)
    return reward,

#  in action select:  Try different actions, get new obs for each, for each obs calculate reward
# using this knn and then choose action with max reward

