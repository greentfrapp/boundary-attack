from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.datasets import mnist
import time
import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image


def orthogonal_perturbation(delta, prev_sample, target_sample):
	# Generate perturbation
	perturb = np.random.randn(28, 28)
	perturb /= np.linalg.norm(perturb)
	perturb *= delta * np.linalg.norm(target_sample - prev_sample)
	# Project perturbation onto sphere around target
	diff = (target_sample - prev_sample).astype(np.float32)
	diff /= np.linalg.norm(diff)
	perturb -= np.dot(perturb, diff) * diff
	# Check overflow and underflow
	overflow = (prev_sample + perturb) - np.ones_like(prev_sample) * 255.
	perturb -= overflow * (overflow > 0)
	underflow = np.zeros_like(prev_sample) - (prev_sample + perturb)
	perturb += underflow * (underflow > 0)
	return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb /= np.linalg.norm(target_sample - prev_sample)
	perturb *= epsilon
	return perturb

def get_diff(sample_1, sample_2):
	return np.mean((sample_1 - sample_2).astype(np.float32) ** 2)

def get_normal_mse(sample_1, sample_2):
	sample_1 = sample_1.astype(np.float32) / 255.
	sample_2 = sample_2.astype(np.float32) / 255.
	return np.mean((sample_1 - sample_2).astype(np.float32) ** 2)

def sample(label):
	_, (x_test, y_test) = mnist.load_data()
	while True:
		choice = np.random.choice(len(y_test))
		if y_test[choice] == label:
			return x_test[choice]

def draw(sample, name, step=0):
	sample = sample.reshape(28, 28).astype(np.uint8)
	sample = Image.fromarray(sample)
	name += time.strftime('_%Y%m%d_%H%M%S_', datetime.datetime.now().timetuple()) + str(step)
	filepath = os.path.join("server/assets", "{}.png".format(name))
	sample.save(filepath)
	return filepath
