import gym
from stable_baselines3 import PPO, A2C
import os
import time
from pongenv import PongWal


env = PongWal()
env.reset()

<<<<<<< Updated upstream
models_dir = "models/1651356886"
model_num = 62

logdir = f"logs/{int(time.time())}"
=======
models_dir = "models/A2C/1651479787"
model_num = 210000
logdir = "logs/1651479787/A2C_0"
>>>>>>> Stashed changes

model = PPO.load(f"{models_dir}/{model_num}.zip", env=env)

TIMESTAMP = 10000
<<<<<<< Updated upstream
for i in range(1, 100000000000): 
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{model_num + i * TIMESTAMP}")
=======
for i in range(1, 1000000000): 
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{(model_num+TIMESTAMP)}")


''' episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample()) '''
env.close()
>>>>>>> Stashed changes
