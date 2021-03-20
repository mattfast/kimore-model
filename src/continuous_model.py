import copy

from sklearn.linear_model import LinearRegression

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.constraints import max_norm

fcnn_default_params = {'loss_function': 'mean_squared_error', 
					   'max_norm_1': 10000, 
					   'max_norm_2': 10,
					   'lr': 0.00005,
					   'epochs_1': 300,
					   'epochs_2': 150}

lstm_default_params = {'loss_function': 'mean_squared_error',
					   'lr': 0.005,
             #'lr': 0.000001,
					   'epochs_1': 300,
					   'epochs_2': 150}

class ContinuousModel(Model):

	def __init__(self, training_data, model_type, target_score='cTS'):
		super().__init__(training_data, model_type, target_score)

	def train_linear(self):

		X_train, y_train = self.preprocess_training_data()

		print("Fitting Linear Regression")
		self.model = LinearRegression().fit(X_train,y_train)
		print("Model Successfully Fit")

	def predict_linear(self, eval_data):

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
		model.add(layers.Dense(1, activation='linear', kernel_constraint=max_norm(params['max_norm_2'])))
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


	def train_lstm(self, params):

		position_train, dct_train, y_train = self.preprocess_training_data()
		num_examples = len(position_train)
		position_len = len(position_train[0][0])
		dct_len = len(dct_train[0])

		# Padding
		print("Padding Training Data (for variable-length input)")
		position_train_pad, max_seq_len, special_value = self.pad(position_train)

		print("Building Model")
		position_input = keras.Input(shape=(None,position_len))
		lstm = layers.Masking(mask_value=special_value)(position_input)
		lstm = layers.BatchNormalization()(lstm)
		lstm = layers.LSTM(128, kernel_constraint=max_norm(50))(lstm)
		#lstm = layers.LSTM(128, return_sequences=True)(lstm)
		#lstm = layers.LSTM(64, kernel_constraint=max_norm(10))(lstm)
		lstm = keras.Model(inputs=position_input, outputs=lstm)

		dct_input = keras.Input(shape=(dct_len,))
		dct_normalized = layers.BatchNormalization()(dct_input)
		dct = keras.Model(inputs=dct_input, outputs=dct_normalized)

		combined = layers.concatenate([lstm.output, dct.output])
		prediction = layers.Dense(1, activation="linear", kernel_constraint=max_norm(10))(combined)
		model = keras.Model(inputs=[lstm.input,dct.input],outputs=prediction)
		model.summary()

		optimizer = keras.optimizers.Adam(lr=params['lr'])
		model.compile(loss=params['loss_function'], optimizer=optimizer)

		print("Fitting Model")
		model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=150, batch_size=num_examples)
		keras.backend.set_value(model.optimizer.learning_rate, params['lr']/2)
		model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=150, batch_size=num_examples)
		#keras.backend.set_value(model.optimizer.learning_rate, params['lr']/8)
		#model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=300, batch_size=num_examples)
		#keras.backend.set_value(model.optimizer.learning_rate, params['lr']/16)
		#model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=params['epochs_2'], batch_size=num_examples)
		self.model = model
		print("Model Successfully Fit")

	def predict_lstm(self, eval_data):

		position_eval, dct_eval = self.preprocess_eval_data(copy.deepcopy(eval_data))

		# Padding
		print("Padding Eval Data (for variable-length input)")
		position_eval_pad, _, _ = self.pad(position_eval)
  
		print("Predicting Quality Score(s)")
		return self.model.predict([position_eval_pad, dct_eval])