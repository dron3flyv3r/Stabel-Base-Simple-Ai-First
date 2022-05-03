import time
from collections import deque
from random import random, randrange

import numpy as np
import pygame
import gym
from gym import spaces

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

POINT_GOL = 30

class PongWal(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PongWal, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(6+POINT_GOL,), dtype=np.float32)

    def step(self, action):
        self.prev_actions.append(action)

        def drawrect(screen, x,y):
            if x <= 0:
                x = 0
            if x >= 699:
                x = 699    
            pygame.draw.rect(screen,RED,[x,y,100,20])

            if action == 0:
                self.rect_change_x = -6
            elif action == 1:
                self.rect_change_x = 6         
            elif action == 2:
                self.rect_change_x = 0 

        self.screen.fill(BLACK)
        self.rect_x += self.rect_change_x
        self.rect_y += self.rect_change_y
        
        self.ball_x += self.ball_change_x
        self.ball_y += self.ball_change_y

    
        #this handles the movement of the ball.
        if self.ball_x<0:
            self.ball_x=0
            self.ball_change_x = self.ball_change_x * -1
        elif self.ball_x>785:
            self.ball_x=785
            self.ball_change_x = self.ball_change_x * -1
        elif self.ball_y<0:
            self.ball_y=0
            self.ball_change_y = self.ball_change_y * -1
        elif self.ball_x>self.rect_x and self.ball_x<self.rect_x+100 and self.ball_y==565:
            self.ball_change_y = self.ball_change_y * -1
            self.score = self.score + 1
        elif self.ball_y>600:
            self.ball_change_y = self.ball_change_y * -1
            self.score = 0    
            self.done = True                    
        pygame.draw.rect(self.screen,WHITE,[self.ball_x,self.ball_y,15,15])
        
        #drawball(screen,ball_x,ball_y)
        drawrect(self.screen,self.rect_x,self.rect_y)
        
        #score board
        self.font= pygame.font.SysFont('Calibri', 15, False, False)
        self.text = self.font.render("Score = " + str(self.score), True, WHITE)
        self.screen.blit(self.text,[600,100])    
        
        pygame.display.flip()  

        self.reward_total = (((abs(self.ball_x - self.rect_x)*-1) + 300)*0.1 + self.score * 100)
        self.reward = self.reward_total - self.prev_reward
        self.prev_reward = self.reward_total

        if self.done:
            self.reward = -100

        self.info = {}

        observation = [self.rect_x, self.rect_change_x, self.ball_x, self.ball_y, self.ball_change_x, self.ball_change_y] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, self.info

    def reset(self):
        pygame.init()

        #Initializing the display window
        self.size = (800,600)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("pong A2C")

        #Starting coordinates of the paddle
        self.rect_x = randrange(50, 750)
        self.rect_y = 580

        #Make the game run
        self.done = False
        self.prev_reward = 0

        #initial speed of the paddle
        self.rect_change_x = 0
        self.rect_change_y = 0

        #initial position of the ball
        self.ball_x = randrange(50, 750)
        self.ball_y = 50

        #speed of the ball
        self.ball_change_x = 5
        self.ball_change_y = 5

        self.score = 0

        self.prev_actions = deque(maxlen = POINT_GOL)  # however long we aspire the snake to be
        for i in range(POINT_GOL):
            self.prev_actions.append(-1) # to create history

        observation = [self.rect_x, self.rect_change_x, self.ball_x, self.ball_y, self.ball_change_x, self.ball_change_y] + list(self.prev_actions)
        observation = np.array(observation)

        return observation  # reward, done, info can't be included
    