import gym

from pongenv import PongWal
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = PongWal()


model = PPO.load("models/PPO/1651496396/2900000.zip")

for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    
    
