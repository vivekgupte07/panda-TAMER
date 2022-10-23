import random
import gym
import gym_panda
from action_select_knn import knn_action_select
from action_select import action_select

env = gym.make("panda-v0")
env.reset()
action_space = [1, 2, 3, 5, 6, 7]
dones = False
gamma = 0
while True:
    if dones:
        print(counter1, counter2)
        obs = env.reset()
        dones = False
    action = 7
    counter1 = 0
    counter2 = 0
    while not dones:
        env.render()
        counter1 +=1
        obs, reward, dones, info = env.step(action)
        g = random.uniform(0, 1)
        prev_action = action
        if g > gamma:
            action = action_select(obs)
            counter2 +=1
            # action = knn_action_select(obs)
        else:
            action = random.choice(action_space)
