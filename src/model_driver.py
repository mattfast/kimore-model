from load_kimore import load_kimore_data

from sklearn.model_selection import train_test_split


class ModelDriver()

	def __init__(self, path, model_type):

		kimore = load_kimore_data(path)

		ex1 = [ex for ex in kimore if ex["Exercise"] == 1]
		ex2 = [ex for ex in kimore if ex["Exercise"] == 2]
		ex3 = [ex for ex in kimore if ex["Exercise"] == 3]
		ex4 = [ex for ex in kimore if ex["Exercise"] == 4]
		ex5 = [ex for ex in kimore if ex["Exercise"] == 5]

		exs = [ex1, ex2, ex3, ex4, ex5]

		for ex in exs:
			X_train, X_test, _, _ = train_test_split(ex, np.zeros(shape=(len(ex))), test_size=0.2, random_state=1)
			X_train, X_val, _, _ = train_test_split(X_train, np.zeros(shape=(len(X_train))), test_size=0.25, random_state=1)

			
			val = Validation(X_train, X_val, model_type, )
			params = val.