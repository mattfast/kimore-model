from model import Model
import math

class Evaluator:
	
	def __init__(self, training_data, eval_data, model_type, target_score='cTS', loss_function='mean_squared_error'):

		self.model = Model(training_data, model_type, target_score=target_score, loss_function=loss_function).train()
		self.predictions, self.indices = self.model.predict(eval_data)
		self.eval_data = eval_data

	def MSE(self):
		
		mse = 0
		for i in range(len(self.indices)):
			true_label = float(self.eval_data[indices[i]])
			mse += (self.predictions[i] - true_label)

		mse /= len(self.indices)

		return mse

	def RMSE(self):
		return math.sqrt(rmse)

	def MAE(self):

		mae = 0
		for i in range(len(self.indices)):
			true_label = float(self.eval_data[indices[i]])
			mae += math.abs(self.predictions[i] - true_label)

		mae /= len(self.indices)

		return mae




