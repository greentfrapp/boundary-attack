from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K
import argparse


class MNISTClassifier(object):
	def __init__(self):
		self.sample_height = 28
		self.sample_width = 28
		return None

	def create_model(self):
		classifier = Sequential()
		classifier.add(Conv2D(
			filters=32,
			kernel_size=(5, 5),
			activation='relu',
			data_format='channels_last',
			input_shape=(self.sample_height, self.sample_width, 1)))
		classifier.add(MaxPooling2D(
			pool_size=(2, 2),
			strides=(2, 2),
			data_format='channels_last'))
		classifier.add(Conv2D(
			filters=64,
			kernel_size=(5, 5),
			activation='relu',
			data_format='channels_last'))
		classifier.add(MaxPooling2D(
			pool_size=(2, 2),
			strides=(2, 2),
			data_format='channels_last'))
		classifier.add(Flatten())
		classifier.add(Dense(
			units=1024,
			activation='relu'))
		classifier.add(Dropout(
			rate=0.4))
		classifier.add(Dense(
			units=10,
			activation='softmax'))
		classifier.compile(
			optimizer=Adam(lr=1e-3),
			loss='categorical_crossentropy')
		return classifier

	def train(self):
		self.classifier = self.create_model()
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train.reshape(-1, 28, 28, 1)
		x_test = x_test.reshape(-1, 28, 28, 1)
		y_train = to_categorical(y_train, num_classes=10)
		y_test = to_categorical(y_test, num_classes=10)
		callbacks = [EarlyStopping(
			monitor='val_loss',
			min_delta=1e-6,
			patience=10)]
		self.classifier.fit(
			x=x_train, 
			y=y_train,
			batch_size=100,
			epochs=50,
			validation_split=0.1,
			callbacks=callbacks)
		self.classifier.save('classifier.h5')
		predictions = self.classifier.predict(x_test)
		y_pred = K.constant(predictions)
		y_true = K.constant(y_test)
		accuracy = np.mean(K.eval(categorical_accuracy(y_true, y_pred)))
		print(accuracy)
		return None

	def load(self):
		self.classifier = load_model('./server/controllers/classifier.h5')
		return None

	def get_confidence(self, sample):
		prediction = self.classifier.predict(sample)
		return prediction

	def predict(self, sample):
		prediction = self.classifier.predict(sample)
		return np.argmax(prediction, axis=1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='MNIST Classifier with Keras')
	parser.add_argument('--train', action='store_true', help='Train a model')
	args = parser.parse_args()
	if args.train:
		model = MNISTClassifier()
		model.train()
	else:
		parser.print_help()
