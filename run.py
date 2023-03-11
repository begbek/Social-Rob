"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gym

from tamer.agent import Tamer


async def main():
	#env_name = "CartPole-v1"
	env_name = 'MountainCar-v0'
	env = gym.make(env_name)

	# hyperparameters
	discount_factor = 1
	epsilon = 0  # vanilla Q learning actually works well with no random exploration
	min_eps = 0
	num_episodes = 10
	tame = False  # set to false for vanilla Q learning
	imper = True # sets the agent to change its action imperatively via human feedback

	# set a timestep for training TAMER
	# the more time per step, the easier for the human
	# but the longer it takes to train (in real time)
	# 0.2 seconds is fast but doable
	tamer_training_timestep = 0.5

	agent = Tamer(env, num_episodes, discount_factor, epsilon, min_eps, tame,
				tamer_training_timestep, model_file_to_load=None, env_name=env_name, imper=imper)

	await agent.train(model_file_to_save='autosave')
	agent.play(n_episodes=1, render=True)
	agent.evaluate(n_episodes=30)
	
	print('coucou')


if __name__ == '__main__':
	asyncio.run(main())
	





