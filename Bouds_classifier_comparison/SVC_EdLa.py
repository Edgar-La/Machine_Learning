import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import matplotlib.pyplot as plt



def run_SVC(X, y_label, h):
	svc = SVC(gamma=0.1, C=10)
	svc.fit(X, y_label)

	#h = .09
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	A = svc.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	return xx, yy, Z
