import numpy as np
import sklearn.metrics as skl
from statistics import mean
#Help from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
from sklearn.model_selection import KFold


#confusion_matrix = skl.confusion_matrix(labels_2, predicted_labels)


def get_ACC(confusion_matrix):
	diag = 0
	for i in range(len(confusion_matrix)):
	  diag += confusion_matrix[i][i]
	ACC = diag/len(labels_2)


k = 4

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 1, 2, 0])
kf = KFold(n_splits=k)
print(kf.get_n_splits(X)); print('\n')


print(kf); print('\n')

KFold(n_splits=4, random_state=None, shuffle=False)
for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]