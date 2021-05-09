import numpy as np
from sklearn.svm import SVC


#This method is call by the MAIN SCRIPT
def run_SVC(X, y_label, xx, yy, Gamma = 0.1, c = 10):
	svc = SVC(gamma = Gamma, C = c)
	svc.fit(X, y_label)

	A = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	print(np.c_[xx.ravel(), yy.ravel()])
	print(len(np.c_[xx.ravel(), yy.ravel()]))
	print(len(A))
	print(len(xx.shape))
	print(xx.shape); print('\n')
	print(len(Z))
	return Z
