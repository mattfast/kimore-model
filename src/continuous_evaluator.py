from continuous_model import ContinuousModel

import math

class ContinuousEvaluator():
	
	def __init__(self, training_data, eval_data, model_type, params, target_score='cTS'):

		self.model = ContinuousModel(training_data, model_type, target_score=target_score)
		self.model.train(params)

		if model_type == 'lstm' or model_type == 'gru':
			self.predictions = self.model.predict(eval_data)
			self.indices = range(len(eval_data))
		else:
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

	# spelman's rank correlation formula
	# https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php
	def rank_correlation(self):

		true_mean = 0
		predicted_mean = 0
		for i in range(len(self.predictions)):
			if math.isnan(self.true_labels[i]):
				nans += 1
			else:
				true_mean += self.true_labels[i]
				predicted_mean += self.predictions[i]

		true_mean /= (len(self.true_labels) - nans)
		predicted_mean /= (len(self.predictions) - nans)

		numerator = 0
		denominator_x = 0
		denominator_y = 0
		for i in range(len(self.predictions)):
			if not math.isnan(self.true_labels[i]):
				denominator_x += (self.predictions[i] - predicted_mean)**2
				denominator_y += (self.true_labels[i] - true_mean)**2

				numerator += (self.predictions[i] - predicted_mean) * (self.true_labels[i] - true_mean)

		denominator = sqrt(denominator_x * denominator_y)

		return (numerator / denominator)

	def get_predictions(self):
		return self.predictions, self.indices

	def get_true_labels(self):
		return self.true_labels

		