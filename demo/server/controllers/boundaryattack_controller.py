import connexion
import six

from server.models.init_labels import InitLabels  # noqa: E501
from server.models.init_response import InitResponse  # noqa: E501
from server.models.step_response import StepResponse  # noqa: E501
from server import util

import numpy as np
import os
import tensorflow.compat.v1 as tf

from server.controllers.MNISTClassifier import MNISTClassifier
import server.controllers.util as ba


def init(initLabels):  # noqa: E501
	"""Initializes a demo

	Initializes a demo # noqa: E501

	:param initLabels: Initialization labels
	:type initLabels: dict | bytes

	:rtype: InitResponse
	"""
	if connexion.request.is_json:
		initLabels = InitLabels.from_dict(connexion.request.get_json())  # noqa: E501
	
	global _attack_class, _target_class, _initial_sample, _adversarial_sample, _target_sample, _n_steps, _delta, _epsilon, _initial_image_loc, _final_image_loc, _score

	_attack_class = initLabels.initial
	_target_class = initLabels.target

	_initial_sample = ba.sample(_attack_class)
	_adversarial_sample = _initial_sample
	_target_sample = ba.sample(_target_class)
	_n_steps = 0
	_delta = 0.5
	_epsilon = 1.
	_score = 0.

	if _initial_image_loc is not None:
		os.system("rm {}".format(_initial_image_loc))
	if _final_image_loc is not None:
		os.system("rm {}".format(_final_image_loc))

	output = InitResponse()
	_initial_image_loc = ba.draw(np.copy(_initial_sample), "initial")
	_final_image_loc = ba.draw(np.copy(_target_sample), "target")
	output.initial_image = _initial_image_loc
	output.final_image = _final_image_loc
	with _sess.as_default():
		with _sess.graph.as_default():
			scores = _classifier.get_confidence(_initial_sample.reshape(1, 28, 28, 1)).astype(float).tolist()[0]
	scores = list(zip(scores, range(10)))
	scores.sort(reverse=True)
	output.scores = scores[:3]
	output.mse = float(ba.get_normal_mse(np.copy(_initial_sample), np.copy(_target_sample)))
	output.step = _n_steps

	return output


def step():  # noqa: E501
	"""Takes a step in the attack

	Takes a step in the attack # noqa: E501


	:rtype: StepResponse
	"""

	global _attack_class, _target_class, _delta, _epsilon, _adversarial_sample, _n_steps, _prev_sample_loc, _score

	with _sess.as_default():
		with _sess.graph.as_default():
			if _n_steps == 0:
				_epsilon = np.linalg.norm((_adversarial_sample - _target_sample).astype(np.float32))
				while True:
					trial_sample = _adversarial_sample + ba.forward_perturbation(_epsilon * np.linalg.norm(_adversarial_sample - _target_sample), _adversarial_sample, _target_sample)
					prediction = _classifier.predict(trial_sample.reshape(1, 28, 28, 1))
					if prediction == _attack_class:
						_adversarial_sample = trial_sample
						break
					else:
						_epsilon *= 0.9
			else:
				while True:
					trial_samples = []
					for i in np.arange(10):
						trial_sample = _adversarial_sample + ba.orthogonal_perturbation(_delta, _adversarial_sample, _target_sample)
						trial_samples.append(trial_sample)
					predictions = _classifier.predict(np.array(trial_samples).reshape(-1, 28, 28, 1))
					d_score = np.mean(predictions == _attack_class)
					if d_score > 0.0:
						if d_score < 0.3:
							_delta *= 0.9
						elif d_score > 0.6:
							_delta /= 0.9
						_adversarial_sample = np.array(trial_samples)[np.where(predictions == _attack_class)[0][0]]
						break
					else:
						_delta *= 0.9
				_epsilon = np.linalg.norm((_adversarial_sample - _target_sample).astype(np.float32))
				while True:
					trial_sample = _adversarial_sample + ba.forward_perturbation(_epsilon, _adversarial_sample, _target_sample)
					prediction = _classifier.predict(trial_sample.reshape(1, 28, 28, 1))
					if prediction == _attack_class:
						_adversarial_sample = trial_sample
						break
					else:
						_epsilon *= 0.9

			_n_steps += 1
			output = StepResponse()
			sample_loc = ba.draw(np.copy(_adversarial_sample), "adversarial", _n_steps)
			output.sample = sample_loc
			scores = _classifier.get_confidence(np.array(_adversarial_sample).reshape(1, 28, 28, 1)).astype(float).tolist()[0]
			scores = list(zip(scores, range(10)))
			scores.sort(reverse=True)
			output.scores = scores[:3]
			output.mse = float(ba.get_normal_mse(np.copy(_adversarial_sample), np.copy(_target_sample)))
			output.step = _n_steps

			if _prev_sample_loc is not None:
				os.system("rm {}".format(_prev_sample_loc))
			_prev_sample_loc = sample_loc
			print(_delta)

	return output

_sess = tf.Session()
_classifier = MNISTClassifier()
with _sess.as_default():
	with _sess.graph.as_default():
		_classifier.load()
_prev_sample_loc = None
_initial_image_loc = None
_final_image_loc = None

# Delete all files in assets other than default folder
for file in os.listdir("server/assets/"):
	if file != "default.png":
		os.system("rm {}".format(os.path.join("server/assets", file)))
