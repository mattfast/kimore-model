from numpy.random import uniform

max_count = 100
fcnn_params = ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']
lstm_params = ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']
gru_params = ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']

class Validation():
	def __init__(self, training_data, validation_data, model_type, set_params, variable_params, scale):

		self.validate_params(set_params, variable_params, model_type)

		self.lowest_mse = float("inf")
		self.lowest_mse_params = {}
		self.params_list = []
		self.mse_list = []

		for i in range(max_count):

			params = {}
			for param in set_params.keys():
				params[param] = set_params[param]

			for param in variable_params.keys():
				value_range = variable_params[param]
				if scale == 'log':
					params[param] = 10**uniform(value_range[0], value_range[1])
				elif scale == 'linear':
					params[param] = uniform(value_range[0], value_range[1])
				else:
					raise ValueError("Scale must be 'log' or 'linear'")

			print("Params: ")
			print(params)
			evaluator = ContinuousEvaluator(training_data, validation_data, model_type, params)

			if evaluator.MSE() < lowest_mse:
				lowest_mse = evaluator.MSE()
				lowest_mse_params = params
				print("New Minimum!")

			print("MSE: ")
			print(mse)
			self.params_list.append(params)
			self.mse_list.append(evaluator.MSE())

	def get_best_params(self):
		return self.lowest_mse_params

	def get_params_list(self):
		return self.params_list

	def get_mse_list(self):
		return self.mse_list

	def validate_params(self, set_params, variable_params, model_type):

		if model_type == 'fcnn':
			params_list = copy.deepcopy(fcnn_params)
		#elif model_type == 'lstm':
		#	params_list = copy.deepcopy(lstm_params)

		for param in set_params.keys():
			if param not in params_list:
				raise ValueError("Params list must contain the following values: ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']")
			else:
				params_list.remove(param)

		for param in variable_params.keys():
			if param not in params_list:
				raise ValueError("Params list must contain the following values: ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']")
			else:
				params_list.remove(param)

		if len(fcnn_params_copy) != 0:
			raise ValueError("Params list must contain the following values: ['loss_function', 'max_norm_1', 'max_norm_2', 'lr', 'epochs_1', 'epochs_2']")
