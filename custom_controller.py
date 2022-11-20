import random
import gym
import gym_panda
from numpy import savetxt
from action_select_knn import knn_action_select
from action_select import action_select
from action_select_nn import nn_action_select
from rewardmodel_classification import train_model
import os


training = False

env = gym.make("panda-v0")
action_space = [0, 1, 2, 3, 4]
dones = False
gamma = 0
env.reset()
f = 0
k = 2
checkpoint = 50
while True:
    if dones:
        print(counter1, counter2)
        obs = env.reset()
        dones = False

    if f >= 50 and f%25==0:
        checkpoint = 25 * k
        k += 1/3

    action = 2
    counter1 = 0
    counter2 = 0
    print(f"Next Checkpoint: {checkpoint}")
    while not dones:
        print(action)
        counter1 += 1
        obs, reward, dones, info = env.step(action)
        env.render()
        f = info['ep_count']
        g = random.uniform(0, 1)
        prev_action = action
        if g >= gamma:

            action = nn_action_select(obs)
            counter2 += 1
        else:
            action = random.choice(action_space)