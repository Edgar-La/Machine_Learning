import numpy as np
from sklearn.neighbors import KNeighborsClassifier


#This method is call by the MAIN SCRIPT
def run_kNN(X, y_label, xx, yy, kNeighbors = 5):
	kNN = KNeighborsClassifier(n_neighbors = kNeighbors)
	kNN.fit(X, y_label)

	A = kNN.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	return Z
