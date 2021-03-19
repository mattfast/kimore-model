from discrete_model import DiscreteModel

import math
import numpy as np

class DiscreteEvaluator():
	
	def __init__(self, training_data, eval_data, model_type, cutoffs, target_score='cTS'):

		self.model = DiscreteModel(training_data, model_type, cutoffs, target_score=target_score)
		self.model.train(params)

		self.predictions, self.indices = self.model.predict(eval_data)
		self.eval_data = eval_data
		self.num_classes = len(self.cutoffs) + 1

		continuous_labels = []
		for i in range(len(self.indices)):
			try:
				x = float(self.eval_data[self.indices[i]][target_score])
			except ValueError:
				x = float("NaN")

			continuous_labels.append(x)

		self.true_labels = self.model.discretize(continuous_labels)

	def accuracy(self):

		accuracy = 0
		nans = 0
		for i in range(len(self.predictions)):
			if math.isnan(self.true_labels[i]):
				nans += 1
			elif self.true_labels[i] == self.predictions[i]:
				accuracy += 1

		accuracy /= len(self.predictions) - nans

		return accuracy

	def confusion_matrix(self):

		confusion_matrix = np.zeros(shape=(0,self.num_classes), dtype='int')

		for i in range(self.num_classes):

			row = np.zeros(shape=(self.num_classes,))
			for j in range(len(self.predictions)):
				if self.true_labels[j] == i:
					row[self.predictions[j]] += 1

			confusion_matrix = np.vstack([confusion_matrix, row])

		return confusion_matrix

	def precision(self):

		confusion_matrix = self.confusion_matrix()
		precision = []
		for i in range(self.num_classes):
			precision.append(confusion_matrix[i,i] / np.sum(confusion_matrix[:,i]))

		return np.array(precision)

	def recall(self):

		confusion_matrix = self.confusion_matrix()
		recall = []
		for i in range(self.num_classes):
			recall.append(confusion_matrix[i,i] / np.sum(confusion_matrix[i,:]))

		return np.array(recall)

	def f1(self):
		precision = self.precision()
		recall = self.recall()

		return (2 * precision * recall) / (precision + recall)
		

