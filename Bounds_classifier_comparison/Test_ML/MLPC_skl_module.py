import numpy as np
from sklearn.neural_network import MLPClassifier

#This method is call by the MAIN SCRIPT
def run_MLPC_skl(X, y_label, xx, yy, Epochs=1):
	clf = MLPClassifier(random_state=1, max_iter=Epochs)
	clf.fit(X, y_label)

	A = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = A.reshape(xx.shape)

	return Z