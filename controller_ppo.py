import gym
import gym_panda
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os


models_dir = "PPO/Trial_27-PPOvTAMER_LF"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = make_vec_env("panda-v0", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
env.reset()

TIMESTEPS = 100000
i = 1
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_27_PPOLFCOMP")
    print("Saving...")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1
