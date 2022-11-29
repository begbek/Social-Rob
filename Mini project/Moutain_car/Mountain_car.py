import gym

import gym.envs.classic_control.mountain_car as MC

env = gym.make("MountainCar-v0")

CAR = MC.MountainCarEnv(env)

CAR.render()
