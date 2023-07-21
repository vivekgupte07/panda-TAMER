import os
import gym
import gym_panda
from stable_baselines3 import DDPG
from stable_baselines3 import A2C, PPO, TD3

if os.path.exists('artifacts/features.csv'):
    os.remove('artifacts/features.csv')
if os.path.exists('artifacts/rewards.csv'):
    os.remove('artifacts/rewards.csv')
env = gym.make("panda-v0")

dones = False
model = PPO.load("models/PPO/TAM_R1_H/10.zip")
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    if dones:
        obs = env.reset()
    env.render()
