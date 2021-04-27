import numpy as np
from sklearn.neighbors import KNeighborsClassifier



def run_kNN(X, y_label, xx, yy, kNeighbors = 5):
	kNN = KNeighborsClassifier(n_neighbors = kNeighbors)
	kNN.fit(X, y_label)

	A = kNN.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)
	print('kNN:'); print('\n')
	print(kNN); print('\n')

	print('A:'); print(len(A)); print('\n')
	print(A); print('\n')

	print('Z:'); print(len(Z)); print('\n')
	print(Z); print('\n')

	print('y\'s:'); print(len(y_label)); print('\n')
	print(y_label); print('\n')	

	return Z
