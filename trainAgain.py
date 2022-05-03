import gym
from stable_baselines3 import PPO, A2C
import os
import time
from pongenv import PongWal


env = PongWal()
env.reset()

models_dir = "models/1651356886"
model_num = 62

logdir = f"logs/{int(time.time())}"

model = PPO.load(f"{models_dir}/{model_num}.zip", env=env)

TIMESTAMP = 10000
for i in range(1, 100000000000): 
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{model_num + i * TIMESTAMP}")
