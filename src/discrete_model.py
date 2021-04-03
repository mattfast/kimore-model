from model import Model
import copy

import math
import numpy as np 
from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical

fcnn_default_params_discrete = {'loss_function': 'categorical_crossentropy', 
                                'lr': 0.0000005,
                                'kernel_l2': 0.001,
                                'bias_l2': 0.001, 
                                'epochs': 500}

lstm_default_params_discrete = {'loss_function': 'categorical_crossentropy', 
                                'lr': 0.00002,
                                'kernel_l2': 0.01,
                                'bias_l2': 0.01,
                                'decay_rate': 0.3,
                                'epochs': 500}

gru_default_params_discrete = {'loss_function': 'categorical_crossentropy', 
                               'kernel_l2': 0.01,
                               'bias_l2': 0.01,
                               'lr': 0.00005,
                               'epochs': 500}

class DiscreteModel(Model):

  def __init__(self, training_data, model_type, target_score='Critiques'):
      super().__init__(training_data, model_type, target_score)
      try:
        self.num_categories = len(self.training_data[0][self.target_score])
      except KeyError:
        self.num_categories = len(self.training_data[1][self.target_score])
  
  def preprocess_training_data_variable(self):
  
      position_train = []
      dct_train = []
      y_train = []
  
      print("Preprocessing Training Data")
      for i in range(len(self.training_data)):
          position_vect, dct_vect = self.preprocess_exercise(self.training_data[i])
          try:
              y_i = self.training_data[i][self.target_score]
          except KeyError:
              print(str(i) + " was degenerate (no score)")
              continue
  
          position_train.append(position_vect)
          dct_train.append(dct_vect)
          y_train.append(y_i)
  
          print(str(i) + "/" + str(len(self.training_data) - 1))
  
      # convert to np arrays
      y_train = np.array(y_train, dtype='int')
      dct_train = np.array(dct_train, dtype='float64')
  
      return position_train, dct_train, y_train
  
  def preprocess_training_data_fixed(self):
  
      first_feature_vect = self.preprocess_exercise(self.training_data[0])
      feature_vect_len = len(first_feature_vect)
  
      X_train = np.zeros(shape=(0,feature_vect_len), dtype='float64')
      y_train = np.zeros(shape=(0,self.num_categories), dtype='int')
  
      print("Preprocessing Training Data")
      for i in range(len(self.training_data)):
          feature_vect = self.preprocess_exercise(self.training_data[i])
  
          try:
              y_i = np.array(self.training_data[i][self.target_score], dtype='int')
          except KeyError:
              print(str(i) + " was degenerate (no score)")
              continue
  
          if len(feature_vect) == feature_vect_len and len(y_i) == self.num_categories:
              X_train = np.vstack([X_train, feature_vect])
              y_train = np.vstack([y_train, y_i])
          elif len(feature_vect) != feature_vect_len:
              print(str(i) + " was degenerate (wrong feature vect len)")
          else:
              print(str(i) + " was degenerate (wrong critiques len)")
  
          print(str(i) + "/" + str(len(self.training_data) - 1))
  
      return X_train, y_train
  
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
  
  def train_fcnn(self, params=fcnn_default_params_discrete):
  
      X_train, y_train = self.preprocess_training_data()
      input_layer_size = len(X_train[0])
  
      model = models.Sequential()
      model.add(keras.Input(shape=(input_layer_size,)))
      model.add(layers.BatchNormalization())
      model.add(layers.Dense(3000, activation='relu', 
                             kernel_regularizer=regularizers.L2(l2=params['kernel_l2']), 
                             bias_regularizer=regularizers.L2(l2=params['bias_l2'])))
      model.add(layers.BatchNormalization())
      model.add(layers.Dense(3000, activation='relu', 
                             kernel_regularizer=regularizers.L2(l2=params['kernel_l2']), 
                             bias_regularizer=regularizers.L2(l2=params['bias_l2'])))
      model.add(layers.Dense(self.num_categories, activation='sigmoid'))
      model.summary()
  
      optimizer = keras.optimizers.Adam(lr=params['lr'])
      model.compile(loss=params['loss_function'], optimizer=optimizer)
  
      print("Fitting Model")
      model.fit(X_train, y_train, epochs=100, batch_size=len(X_train))
      keras.backend.set_value(model.optimizer.learning_rate, params['lr']/2)
      model.fit(X_train, y_train, epochs=100, batch_size=len(X_train))
      keras.backend.set_value(model.optimizer.learning_rate, params['lr']/4)
      model.fit(X_train, y_train, epochs=500, batch_size=len(X_train))
      self.model = model
      print("Model Successfully Fit")
  
  def predict_fcnn(self, eval_data):
  
      X_eval, indices = self.preprocess_eval_data(copy.deepcopy(eval_data))
  
      print("Predicting Quality Score(s)")
      return self.model.predict(X_eval), indices
  
  
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
      lstm = layers.LSTM(128, 
                         kernel_regularizer=regularizers.L2(l2=params['kernel_l2']), 
                         bias_regularizer=regularizers.L2(params['bias_l2']))(lstm)
      lstm = keras.Model(inputs=position_input, outputs=lstm)
  
      dct_input = keras.Input(shape=(dct_len,))
      dct_normalized = layers.BatchNormalization()(dct_input)
      dct = keras.Model(inputs=dct_input, outputs=dct_normalized)
  
      combined = layers.concatenate([lstm.output, dct.output])
      prediction = layers.Dense(self.num_categories, activation='sigmoid')(combined)
      model = keras.Model(inputs=[lstm.input,dct.input],outputs=prediction)
      model.summary()
      
      lr_schedule = keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=params['lr'],
          decay_steps=100,
          decay_rate=params['decay_rate'])
      optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
      model.compile(loss=params['loss_function'], optimizer=optimizer)
  
      print("Fitting Model")
      model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=500, batch_size=num_examples)
      self.model = model
      print("Model Successfully Fit")
  
  def predict_lstm(self, eval_data):
  
      return self.predict_variable(eval_data)
  
  def train_gru(self, params):
  
      position_train, dct_train, y_train = self.preprocess_training_data()
      num_examples = len(position_train)
      position_len = len(position_train[0][0])
      dct_len = len(dct_train[0])
  
      # Padding
      print("Padding Training Data (for variable-length input)")
      position_train_pad, max_seq_len, special_value = self.pad(position_train)
  
      print("Building Model")
      position_input = keras.Input(shape=(None,position_len))
      gru = layers.Masking(mask_value=special_value)(position_input)
      gru = layers.BatchNormalization()(gru)
      gru = layers.GRU(128, 
                       kernel_regularizer=regularizers.L2(l2=params['kernel_l2']), 
                       bias_regularizer=regularizers.L2(params['bias_l2']))(gru)
      gru = keras.Model(inputs=position_input, outputs=gru)
  
      dct_input = keras.Input(shape=(dct_len,))
      dct_normalized = layers.BatchNormalization()(dct_input)
      dct = keras.Model(inputs=dct_input, outputs=dct_normalized)
  
      combined = layers.concatenate([gru.output, dct.output])
      prediction = layers.Dense(self.num_categories, activation='sigmoid')(combined)
      model = keras.Model(inputs=[gru.input,dct.input],outputs=prediction)
      model.summary()
  
      lr_schedule = keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=params['lr'],
          decay_steps=100,
          decay_rate=params['decay_rate'])
      optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
      model.compile(loss=params['loss_function'], optimizer=optimizer)
  
      print("Fitting Model")
      model.fit(x=[position_train_pad, dct_train], y=y_train, epochs=500, batch_size=num_examples)
      self.model = model
      print("Model Successfully Fit")
  
  def predict_gru(self, eval_data):
  
      return self.predict_variable(eval_data)
  
  def predict_variable(self, eval_data):
  
      position_eval, dct_eval = self.preprocess_eval_data(copy.deepcopy(eval_data))
  
      # Padding
      print("Padding Eval Data (for variable-length input)")
      position_eval_pad, _, _ = self.pad(position_eval)
  
      print("Predicting Quality Score(s)")
      return self.model.predict([position_eval_pad, dct_eval])