import gym
import gym_panda
from stable_baselines3 import A2C, PPO, TD3
from stable_baselines3 import DDPG
import os

if os.path.exists('features.csv'):
    os.remove('features.csv')
if os.path.exists('rewards.csv'):
    os.remove('rewards.csv')
env = gym.make("panda-v0")

dones = False
model = PPO.load("PPO/TAM_R1_H/10.zip")
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    if dones:
        obs = env.reset()
    env.render()
