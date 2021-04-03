from preprocessing import *

import copy
import numpy as np

from tensorflow import keras

class Model():

	def __init__(self, training_data, model_type, target_score):

		print("Copying Training Set")
		self.training_data = copy.deepcopy(training_data)
		self.model_type = model_type
		self.target_score = target_score

		return

	def pad(self, data, special_value=-100):

		num_examples = len(data)
		feature_len = len(data[0][0])

		max_seq_len = 0
		for i in range(num_examples):
			if max_seq_len < len(data[i]):
				max_seq_len = len(data[i])

		data_pad = np.full((num_examples, max_seq_len, feature_len), fill_value=special_value)
		for i in range(num_examples):
			seq_len = len(data[i])
			data_pad[i,:seq_len,:] = data[i]

		return data_pad, max_seq_len, special_value


	def preprocess_exercise(self, exercise_data):

		exercise_data = to_np_arrays(exercise_data)
		exercise_data = averaging_filter(exercise_data)
		exercise_data = center_wcs(exercise_data)
		
		if self.model_type != 'lstm' and self.model_type != 'gru':
			exercise_data = crop(exercise_data, fixed=True)
		else:
			exercise_data = crop(exercise_data)

		dct_dict = dct(exercise_data)

		return create_feature_vect(exercise_data, dct_dict, self.model_type)



	def preprocess_eval_data_variable(self, eval_data):

		position_eval = []
		dct_eval = []

		print("Preprocessing Eval Data:")
		for i in range(len(eval_data)):
			position_vect, dct_vect = self.preprocess_exercise(eval_data[i])
			position_eval.append(position_vect)
			dct_eval.append(dct_vect)

			print(str(i) + "/" + str(len(eval_data) - 1))

		return position_eval, np.array(dct_eval, dtype='float64')


	def preprocess_eval_data_fixed(self, eval_data):

		first_feature_vect = self.preprocess_exercise(eval_data[0])
		feature_vect_len = len(first_feature_vect)

		X_eval = np.zeros(shape=(0,feature_vect_len), dtype='float64')
		indices = []

		print("Preprocessing Eval Data:")
		for i in range(len(eval_data)):
			feature_vect = self.preprocess_exercise(eval_data[i])
			if len(feature_vect) == feature_vect_len:
				X_eval = np.vstack([X_eval, feature_vect])
				indices.append(i)
			else:
				print(str(i) + "was degenerate (wrong feature vect len)")

			print(str(i) + "/" + str(len(eval_data) - 1))

		return X_eval, indices

	def preprocess_training_data(self):
		if self.model_type == 'linear' or self.model_type == 'logistic' or self.model_type == 'fcnn':
			return self.preprocess_training_data_fixed()
		elif self.model_type == 'lstm' or self.model_type == 'gru':
			return self.preprocess_training_data_variable()
		else:
			raise ValueError('Unknown model type')

	def preprocess_eval_data(self, eval_data):
		if self.model_type == 'linear' or self.model_type == 'logistic' or self.model_type == 'fcnn':
			return self.preprocess_eval_data_fixed(eval_data)
		elif self.model_type == 'lstm' or self.model_type == 'gru':
			return self.preprocess_eval_data_variable(eval_data)
		else:
			raise ValueError('Unknown model type')

	def to_plot(self, path):

		if self.model_type == 'linear' or self.model_type == 'logistic':
			raise ValueError("to_plot() can only be applied to keras models")

		keras.utils.plot_model(self.model, to_file=path, show_shapes=True)


	def train(self, params={}):
		if self.model_type == 'linear':
			self.train_linear()
		elif self.model_type == 'logistic':
			self.train_logistic()
		elif self.model_type == 'fcnn':
			self.train_fcnn(params)
		elif self.model_type == 'lstm':
			self.train_lstm(params)
		elif self.model_type == 'gru':
			self.train_gru(params)
		else:
			raise ValueError('Unknown model type')


	def predict(self, eval_data):
		if self.model_type == 'linear':
			return self.predict_linear(eval_data)
		elif self.model_type == 'logistic':
			return self.predict_logistic(eval_data)
		elif self.model_type == 'fcnn':
			return self.predict_fcnn(eval_data)
		elif self.model_type == 'lstm':
			return self.predict_lstm(eval_data)
		elif self.model_type == 'gru':
			return self.predict_gru(eval_data)
		else:
			raise ValueError('Unknown model type')