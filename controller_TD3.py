import gym
import gym_panda
import numpy as np
import os

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

models_dir = "TD3/R3(1)"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("panda-v0")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100000
i = 1
while i <= 10:
    model.learn(total_timesteps=TIMESTEPS, log_interval=1, reset_num_timesteps=False, tb_log_name='R3_TD3(1)')
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1
