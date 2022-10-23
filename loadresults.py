import gym
import gym_panda
from stable_baselines3 import A2C, PPO
from stable_baselines3 import DDPG

env = gym.make("panda-v0")

dones = False
model = PPO.load("PPO/Trial_25-PPOvDDPG/300000.zip")
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
    env.render()

