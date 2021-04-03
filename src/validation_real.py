from sklearn.model_selection import train_test_split
from random import uniform

kimore = add_binary_labels(kimore, 'drive/MyDrive/Princeton/Senior Year/Thesis/Data Collection/kimore_critiques.xlsx')
squats = [ex for ex in kimore if ex["Exercise"] == 5]

X_train, X_test, _, _ = train_test_split(squats, np.zeros(shape=(len(squats))), test_size=0.2, random_state=1)
X_train, X_val, _, _ = train_test_split(X_train, np.zeros(shape=(len(X_train))), test_size=0.25, random_state=1)

variable_params = {'kernel_l2': [-6, -4, 'log'],
                   'bias_l2': [-3, -1, 'log'],
                   'lr': [0.00001, 0.0002, 'linear'],
                   'decay_rate': [0.15, 0.45, 'linear']}

params_list = []
accuracies_list = []
max_count = 10
for i in range(max_count):
    print("ITERATION")
    print(i)
    params = {'loss_function': 'categorical_crossentropy'}
    for param in variable_params.keys():
      value_range = variable_params[param]
      scale = value_range[2]
      if scale == 'log':
        params[param] = 10**uniform(value_range[0], value_range[1])
      elif scale == 'linear':
        params[param] = uniform(value_range[0], value_range[1])
      else:
        raise ValueError("Scale must be 'log' or 'linear'")

    print("Params: ")
    print(params)
    eval = DiscreteEvaluator(X_train, X_val, 'gru', params)

    print("Accuracies: ")
    print(eval.accuracy())
    params_list.append(params)
    accuracies_list.append(eval.accuracy())