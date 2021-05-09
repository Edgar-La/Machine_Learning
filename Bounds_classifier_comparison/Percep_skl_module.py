import numpy as np
from sklearn.linear_model import Perceptron


#This method is call by the MAIN SCRIPT
def run_Percep_skl(X, y_label, xx, yy, L_step = .005):
	clf = Perceptron(tol=L_step, random_state=0)
	clf.fit(X, y_label)

	A = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	return Z
