import numpy as np
from sklearn.svm import SVC



def run_SVC(X, y_label, xx, yy, Gamma = 0.1, c = 10):
	svc = SVC(gamma = Gamma, C = c)
	svc.fit(X, y_label)

	A = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	return Z
