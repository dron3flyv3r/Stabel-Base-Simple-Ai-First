import gym
from stable_baselines3 import PPO, A2C
import os
import time
from pongenv import PongWal
from snakeenv import SnekEnv

#simple makes the path to the folder for trining
models_dir = f"models/PPO/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

#make the folder if they dont exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

#here I set the envirement and reset it to start trining 
env = PongWal()
env.reset()

#this is the model setup
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

#here I train the and save it
TIMESTAMP = 10000
for i in range(1, 1000000000): 
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{i*TIMESTAMP}")
env.close()