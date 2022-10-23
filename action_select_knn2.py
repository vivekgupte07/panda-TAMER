import random
import os
import numpy as np
from numpy import savetxt, loadtxt
from knn_reward import find_neighbours


def knn_action_select(obs, last_action):
    current_feature = obs + last_action
