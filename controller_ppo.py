import gym
import gym_panda
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os


models_dir = "PPO/MATH_Fred"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = make_vec_env("panda-v0", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 50
i = 1
while i <= 10:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="MATH_Fred_PPO")
    print("Saving...")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1
