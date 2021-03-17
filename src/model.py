from preprocessing import *

import copy
import numpy as np

class Model():

	def __init__(self, training_data, model_type, target_score='cTS', loss_function='mean_squared_error'):

		print("Copying Training Set")
		self.training_data = copy.deepcopy(training_data)
		self.model_type = model_type
		self.loss_function = loss_function
		self.target_score = target_score

		return

	def preprocess_data(self, exercise_data):

		exercise_data = to_np_arrays(exercise_data)
		exercise_data = averaging_filter(exercise_data)
		exercise_data = center_wcs(exercise_data)
		
		if self.model_type != 'lstm':
			exercise_data = crop(exercise_data, fixed=True)
		else:
			exercise_data = crop(exercise_data)

		dct_dict = dct(exercise_data)

		return create_feature_vect(exercise_data, dct_dict)


	def preprocess_training_data(self):

		first_feature_vect = self.preprocess_data(self.training_data[0])
		feature_vect_len = len(first_feature_vect)

		X_train = np.zeros(shape=(0,feature_vect_len), dtype='float64')
		y_train = np.zeros(shape=0, dtype='float64')

		print("Preprocessing Training Data")
		for i in range(len(self.training_data)):
			feature_vect = self.preprocess_data(self.training_data[i])

			try:
				y_i = float(self.training_data[i][self.target_score])
			except ValueError:
				print(str(i) + " was degenerate (no score)")
				continue

			if len(feature_vect) == feature_vect_len:
				X_train = np.vstack([X_train, feature_vect])
				y_train = np.append(y_train, y_i)
			else:
				print(str(i) + " was degenerate (wrong feature vect len)")

			print(str(i) + "/" + str(len(self.training_data) - 1))

		return X_train, y_train


	def preprocess_eval_data(self, eval_data):

		first_feature_vect = self.preprocess_data(eval_data[0])
		feature_vect_len = len(first_feature_vect)

		X_eval = np.zeros(shape=(0,feature_vect_len), dtype='float64')
		indices = []

		print("Preprocessing Eval Data:")
		for i in range(len(eval_data)):
			feature_vect = self.preprocess_data(eval_data[i])
			if len(feature_vect) == feature_vect_len:
				X_eval = np.vstack([X_eval, feature_vect])
				indices.append(i)
			else:
				print(str(i) + "was degenerate (wrong feature vect len)")

			print(str(i) + "/" + str(len(eval_data) - 1))

		return X_eval, indices

	def train(self):
		if self.model_type == 'linear':
			self.train_linear()
		elif self.model_type == 'logistic':
			self.train_logistic()
		elif self.model_type == 'fcnn':
			self.train_fcnn()
		#elif self.model_type == 'lstm':
		#	self.train_lstm()
		else:
			raise ValueError('Unknown model type')


	def predict(self, exercise_data):
		if self.model_type == 'linear':
			return self.predict_linear(exercise_data)
		elif self.model_type == 'logistic':
			return self.predict_logistic()
		elif self.model_type == 'fcnn':
			return self.predict_fcnn(exercise_data)
		#elif self.model_type == 'lstm':
		#	self.predict_lstm(exercise_data)
		else:
			raise ValueError('Unknown model type')