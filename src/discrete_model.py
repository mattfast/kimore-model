from model import Model
import copy

import math
import numpy as np
from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

fcnn_default_params = {'loss_function': 'categorical_crossentropy', 
					   'max_norm_1': 10000, 
					   'max_norm_2': 10,
					   'lr': 0.00005,
					   'epochs_1': 300,
					   'epochs_2': 150}

class DiscreteModel(Model):

	def __init__(self, training_data, model_type, cutoffs, target_score='cTS'):
		super().__init__(training_data, model_type, target_score)
		self.cutoffs = cutoffs

	def discretize(self, continuous_data):

		discrete_data = []
		for i in range(len(continuous_data)):

			if math.isnan(continuous_data[i]):
				discrete_data.append(float("NaN"))
				continue

			for j in range(len(self.cutoffs)):
				if continuous_data[i] < self.cutoffs[j]:
					discrete_data.append(j)

			if len(discrete_data[i]) == i:
				discrete_data.append(len(self.cutoffs))

		return np.array(discrete_data)

	def train_logistic(self):

		X_train, y_train_continuous = self.preprocess_training_data()
		y_train = self.discretize(y_train_continuous)

		print("Fitting Linear Regression")
		self.model = LogisticRegression().fit(X_train,y_train)
		print("Model Successfully Fit")

	def predict_logistic(self, eval_data):

		print("Copying Eval Set")
		X_eval, indices = self.preprocess_eval_data(copy.deepcopy(eval_data))

		print("Predicting Quality Score(s)")
		return self.model.predict(X_eval), indices

	def train_fcnn(self, params=fcnn_default_params):

		X_train, y_train = self.preprocess_training_data()
		input_layer_size = len(X_train[0])

		model = models.Sequential()
		model.add(keras.Input(shape=(input_layer_size,)))
		model.add(layers.Dense(3000, activation='relu', kernel_constraint=max_norm(params['max_norm_1'])))
		model.add(layers.Dense(3000, activation='relu', kernel_constraint=max_norm(params['max_norm_1'])))
		model.add(layers.Dense(len(self.cutoffs) + 1, activation='sigmoid', kernel_constraint=max_norm(params['max_norm_2'])))
		model.summary()
  
		optimizer = keras.optimizers.Adam(lr=params['lr'])
		model.compile(loss=params['loss_function'], optimizer=optimizer)

		print("Fitting Model")
		model.fit(X_train, y_train, epochs=params['epochs_1'], batch_size=len(X_train))
		keras.backend.set_value(model.optimizer.learning_rate, params['lr']/2)
		model.fit(X_train, y_train, epochs=params['epochs_2'], batch_size=len(X_train))
		self.model = model
		print("Model Successfully Fit")

	def predict_fcnn(self, eval_data):

		X_eval, indices = self.preprocess_eval_data(copy.deepcopy(eval_data))

		print("Predicting Quality Score(s)")
		return self.model.predict(X_eval), indices

	#def train_lstm(self):

	#def predict_lstm(self):


