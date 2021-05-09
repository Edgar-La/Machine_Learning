#Perceptron_module_EdLa
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class Perceptron(object):
    """
    Simple implementation of the perceptron algorithm
    """

    def __init__(self, w0=1, w1=0.1, w2=0.1):

        # weights
        self.w0 = w0  # bias
        self.w1 = w1
        self.w2 = w2

    def step_function(self, z):
        if z >= 0:
            return 1
        else:
            return 0

    def weighted_sum_inputs(self, x1, x2):
        return sum([1 * self.w0, x1 * self.w1, x2 * self.w2])

    def predict(self, x1, x2):
        """
        Uses the step function to determine the output
        """
        z = self.weighted_sum_inputs(x1, x2)

        return self.step_function(z)

    def predict_boundary(self, x):
        """
        Used to predict the boundaries of our classifier
        """
        return -(self.w1 * x + self.w0) / self.w2

    def fit(self, X, y, epochs=1, step=0.1, verbose=True):
        """
        Train the model given the dataset
        """
        errors = []

        for epoch in range(epochs):
            error = 0
            for i in range(0, len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                # The update is proportional to the step size and the error
                update = step * (target - self.predict(x1, x2))
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)
            '''
            if verbose:
                print('Epochs: {} - Error: {} - Errors from all epochs: {}'\
                      .format(epoch, error, errors))
			'''






def run_Perceptron(X, y_label, xx, yy):
	train_x = pd.DataFrame({'x1':X[:,0],'x2':X[:,1]})
	train_y = pd.DataFrame({'':y_label})


	my_perceptron = Perceptron(0.1,0.1)
	my_perceptron.fit(train_x, train_y, epochs=1, step=0.005)

	data = np.c_[xx.ravel(), yy.ravel()]
	test_x = pd.DataFrame({'x1':data[:,0],'x2':data[:,1]})

	pred_y = np.array(test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2), axis=1))
	Z = pred_y.reshape(xx.shape)
	print(len(pred_y))
	print(len(xx.shape))
	print(xx.shape)
	print(len(Z))
	return Z


from Read_datasets_EdLa import read_datasets
datasets_names = ['Data/dataset_classifiers1.csv',		#Datasets names
					'Data/dataset_classifiers2.csv',
					'Data/dataset_classifiers3.csv']
X, y_label = read_datasets(datasets_names)
	
x_min, x_max = X[0][:, 0].min() - 1, X[0][:, 0].max() + 1
y_min, y_max = X[0][:, 1].min() - 1, X[0][:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .09), np.arange(y_min, y_max, .09))



#Z = run_Perceptron(train_x, train_y, xx, yy)
pred_ = run_Perceptron(X[0], y_label[0], xx, yy)
#print(pred_)