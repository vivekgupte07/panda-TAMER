import gym
import gym_panda
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os

models_dir = "Trial_18/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = make_vec_env("panda-v0", n_envs=16)
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 100000
i = 1
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_18")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1
