import gym
from stable_baselines3 import PPO, A2C
import os
import time
from pongenv import PongWal
from snakeenv import SnekEnv

models_dir = f"models/PPO/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PongWal()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTAMP = 1000
for i in range(1, 1000000000): 
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{i*TIMESTAMP}")
env.close()