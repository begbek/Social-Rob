import os
import pygame
import speech_recognition as sr
from gtts import gTTS
import sys
import numpy as np
import time 

import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from deepface import DeepFace as DF

MOUNTAINCAR_ACTION_IMPER = {'left': 0, 'none': 1, 'right': 2}
CARTPOLE_ACTION_IMPER = {'left': 0, 'right': 1, 'Pole_NONE': 2}

IMPERATIVE_MAP = {"CartPole-v1" : CARTPOLE_ACTION_IMPER, 'MountainCar-v0' : MOUNTAINCAR_ACTION_IMPER}

class Interface:
	""" Pygame interface for training TAMER """

	def __init__(self, action_map, tame=True, imper=False):
		self.action_map = action_map
		pygame.init()
		self.font = pygame.font.Font("freesansbold.ttf", 32)

		# set position of pygame window (so it doesn't overlap with gym)
		os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
		os.environ["SDL_VIDEO_CENTERED"] = "0"

		self.screen = pygame.display.set_mode((200, 100))
		area = self.screen.fill((0, 0, 0))
		pygame.display.update(area)

		self.tame = tame
		self.imper = imper

		self.capture = cv2.VideoCapture(0)
		self.capture.set(3,840)
		self.capture.set(4,640)
		self.detected_face = False
		if self.tame : self.init_deepface()

		self.previous_action = None
		self.emotions_reward = {
			'happy' : 1,
			'neutral' : 0,
			'sad' : -1,
			'angry' : -1,
			'fear' : -1,
			'surprise' : -1,
			'disgust' : -1
		}

	def init_deepface(self) :
		_, img = self.capture.read()
		img = cv2.flip(img,1)

		try :
			result = DF.analyze(img, actions = ['emotion'], enforce_detection=True, prog_bar=False)
			self.detected_face = True

		except ValueError as e :
			print(f"\nFace not detected with enforced detection.\nYou may want to restart with a better exposure and without glasses (if needed).\n\nFace detection will be launched without enforcing.\n\tResults may vary!\n")
			
			stop = input("\nContinue? [Y/n]")
			if stop=='n' : raise("Program stopped early.")

			i = 0
			while i!=3 :
				i+=1
				print(f"{3-i} !")
				time.sleep(0.5)
			self.detected_face = False

	def get_emotional_feedback(self):
		"""
		Get human input. Showing happiness toward the agent's action rewards him with a positive value.
			Neutral rewards him 0 and a negative emotion (sad,angry,disgust) penalize his choice.
			Your face traits should overexagerated in order to get stable result.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""
		time.sleep(0.15)

		_, img = self.capture.read()
		img = cv2.flip(img,1)
		#cv2.imshow('Image',img)
		#cv2.waitKey(1)

		try : result = DF.analyze(img, actions = ['emotion'], enforce_detection=True, prog_bar=False)
		except ValueError as e : 
			print('\nFace not detected. Enforced detection not available. Mind your exposure.')
			result = DF.analyze(img, actions = ['emotion'], enforce_detection=False, prog_bar=False)

		main_emotion = result['dominant_emotion']
		print(main_emotion)

		if self.emotions_reward[main_emotion]==-1:
			print('REWARD -1')
			area = self.screen.fill((255, 0, 0))
			pygame.display.update(area)
			reward = -1

		elif self.emotions_reward[main_emotion]:
			print('REWARD 1')
			area = self.screen.fill((0, 255, 0))
			pygame.display.update(area)
			reward = 1

		else:
			print('REWARD 0')
			reward = 0
		time.sleep(0.05)
		return reward

	def get_scalar_feedback(self):
		"""
		Get human input. 'W' key for positive, 'A' key for negative.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""
		reward = 0
		area = None
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					area = self.screen.fill((0, 255, 0))
					reward = 1
					break
				elif event.key == pygame.K_a:
					area = self.screen.fill((255, 0, 0))
					reward = -1
					break
		pygame.display.update(area)
		return reward

	def get_imperative_feedback(self, action, env_name):
		"""
		Get human input. 'Left' key to choose the action to go left and the 'Right' key for right.
		Returns: action the agent should have choosen
		"""
		new_action = action
		time.sleep(0.3)
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_RIGHT:
					new_action = IMPERATIVE_MAP[env_name]['right']
					break
				elif event.key == pygame.K_LEFT:
					new_action = IMPERATIVE_MAP[env_name]['left']
					break
				elif event.key == pygame.K_DOWN:
					new_action = IMPERATIVE_MAP[env_name]['none']
					break
		
		if new_action != action :
			action = new_action
			area = self.screen.fill((255, 0, 0))
			pygame.display.update(area)
			time.sleep(0.05)
		return action

	def show_action(self, action):
		"""
		Show agent's action on pygame screen
		Args:
			action: numerical action (for MountainCar environment only currently)
		"""

		area = self.screen.fill((0, 0, 0))
		pygame.display.update(area)
		text = self.font.render(self.action_map[action], True, (255, 255, 255))
		text_rect = text.get_rect()
		text_rect.center = (100, 50)
		area = self.screen.blit(text, text_rect)
		pygame.display.update(area)

		if self.previous_action != action : time.sleep(1.0)
		self.previous_action = action
