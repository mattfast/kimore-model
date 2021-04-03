from discrete_model import DiscreteModel

import math
import numpy as np

class DiscreteEvaluator():
	
	def __init__(self, training_data, eval_data, model_type, params, target_score='Critiques'):

		self.model = DiscreteModel(training_data, model_type, target_score=target_score)
		self.model.train(params)

		if model_type == 'lstm' or model_type == 'gru':
			self.predictions = self.model.predict(eval_data)
			self.indices = range(len(eval_data))
		else:
			self.predictions, self.indices = self.model.predict(eval_data)
		self.eval_data = eval_data

		self.num_categories = len(training_data[1][target_score])
		self.true_labels = np.zeros(shape=(0,self.num_categories), dtype='int')
		for i in range(len(self.indices)):
			try:
				label = self.eval_data[self.indices[i]][target_score]
			except KeyError:
				label = np.zeros(shape=(self.num_categories), dtype='int')
				print(str(i) + " was degenerate (no score)")

			self.true_labels = np.vstack([self.true_labels, label])

	def accuracy(self):

		accuracies = np.zeros(shape=(self.num_categories), dtype='float64')
		for i in range(len(self.predictions)):
			for j in range(self.num_categories):
				if self.predictions[i,j] >= 0.5 and self.true_labels[i,j] == 1:
					accuracies[j] += 1
				if self.predictions[i,j] < 0.5 and self.true_labels[i,j] == 0:
					accuracies[j] += 1

		accuracies /= len(self.predictions)

		return accuracies

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
		

