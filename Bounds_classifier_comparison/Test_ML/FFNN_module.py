#Feed Forward Neural Network

import numpy as np
import pandas as pd
#This is for avoid future warning: RuntimeWarning: overflow encountered in exp
from scipy.special import expit

def sigmoid(s):
    # Activation function
    return 1 / (1 + np.exp(-s))


def sigmoid_prime(s):
    # Derivative of the sigmoid
    return sigmoid(s) * (1 - sigmoid(s))


class FFNN(object):

    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        # Adding 1 as it will be our bias
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size

        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        # The whole weight matrix, from the inputs till the hidden layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        # The final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # Forward propagation through our network
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.z1 = np.dot(X, self.w1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = sigmoid(self.z1)  # activation function
        self.z3 = np.dot(self.z2, self.w2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = sigmoid(self.z3)  # final activation function
        return o

    def backward(self, X, y, output, step):
        # Backward propagation of the errors
        X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
        self.o_error = y - output  # error in output
        self.o_delta = self.o_error * sigmoid_prime(output) * step  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(
            self.w2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * sigmoid_prime(self.z2) * step  # applying derivative of sigmoid to z2 error

        self.w1 += X.T.dot(self.z2_delta)  # adjusting first of weights
        self.w2 += self.z2.T.dot(self.o_delta)  # adjusting second set of weights

    def predict(self, X):
        return forward(self, X)

    def fit(self, X, y, epochs=10, step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1  # Adding 1 to the inputs to include the bias in the weight
            output = self.forward(X)
            self.backward(X, y, output, step)


######################################################################

def convert_values(X, y_label):
	train_x = pd.DataFrame(data=X, columns=["x1", "x2"])
	train_y = []
	for n in range(len(y_label)):
		train_y.append([int(y_label[n])])
	train_y = np.array(train_y)
	return train_x, train_y

#train_x, train_y = convert_values(X, y_label)
######################################################################
def get_binary_labels(pred_y):
	pred_y_ = [i[0] for i in pred_y]
	#print(pred_y_)
	threshold = 0.5
	pred_y_binary = [0 if i < threshold else 1 for i in pred_y_]
	return np.array(pred_y_binary)

######################################################################

def run_FFNN(X, y_label, xx, yy, Epochs=1, L_step = .005):
	train_x, train_y = convert_values(X, y_label)
	#print(train_x)
	#print(train_y)

	my_network = FFNN()
	my_network.fit(train_x, train_y, epochs=Epochs, step=L_step)

	
	test_x_ = np.c_[xx.ravel(), yy.ravel()]
	test_x = pd.DataFrame(data=test_x_, columns=["x1", "x2"])

	pred_y = test_x.apply(my_network.forward, axis=1)

	pred_y_binary = get_binary_labels(pred_y)
	Z = pred_y_binary.reshape(xx.shape)
	return Z
	

'''
################ RUNNER
def generate_datasets(datasets_names):
	return np.array(pd.read_csv(datasets_names, index_col = 0))
	

def clean_datasets(datasets):
	X = []
	for n in range(len(datasets)):
		X.append([datasets[n][0],datasets[n][1]])
	X = np.array(X)

	y_label = datasets[:,2]
	return X, y_label


def read_datasets(datasets_names):
	datasets = generate_datasets(datasets_names)
	X, y_label = clean_datasets(datasets)
	return X, y_label

######################################################################

import os; os.system('clear')
X, y_label = read_datasets('Data/dataset_classifiers1.csv')
#print(X); print(len(X))
#print(y_label); print(len(y_label))
#for k in range(len(y_label)):
#print(y_label[k])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .09), np.arange(y_min, y_max, .09))

Z = run_FFNN(X, y_label, xx, yy, Epochs=10000, L_step = .005)

#print(xx); print(len(xx))
#print(yy); print(len(yy))
	#print(Z); print(len(Z))
for k in range(len(Z)):
	print(Z[k])
#print(Z)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
colormap = plt.cm.RdYlBu

plt.pcolormesh(xx, yy, Z, cmap=colormap, shading='auto')
plt.scatter(X[:, 0], X[:, 1], c=y_label, s=7, cmap=cm_bright, edgecolor='k')
plt.show()
'''