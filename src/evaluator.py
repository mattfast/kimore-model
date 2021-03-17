from model import Model
import math

class Evaluator():
	
	def __init__(self, training_data, eval_data, model_type, target_score='cTS', loss_function='mean_squared_error'):

		self.model = Model(training_data, model_type, target_score=target_score, loss_function=loss_function)
		self.model.train()

		self.predictions, self.indices = self.model.predict(eval_data)
		self.eval_data = eval_data

		self.true_labels = []
		for i in range(len(self.indices)):
			try:
				x = float(self.eval_data[self.indices[i]][target_score])
			except ValueError:
				x = float("NaN")

			self.true_labels.append(x)

	def MSE(self):

		mse = 0
		nans = 0
		for i in range(len(self.predictions)):
			if math.isnan(self.true_labels[i]):
				nans += 1
			else:
				mse += (self.predictions[i] - self.true_labels[i])**2

		mse /= (len(self.predictions) - nans)

		return mse

	def RMSE(self):
		return math.sqrt(self.MSE())

	def MAE(self):

		mae = 0
		nans = 0
		for i in range(len(self.predictions)):
			if math.isnan(self.true_labels[i]):
				nans += 1
			else:
				mae += abs(self.predictions[i] - self.true_labels[i])

		mae /= (len(self.predictions) - nans)

		return mae

	def get_predictions(self):
		return self.predictions, self.indices

	def get_true_labels(self):
		return self.true_labels





