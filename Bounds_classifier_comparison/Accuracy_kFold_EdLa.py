import numpy as np
import sklearn.metrics as skl
from statistics import mean
#Help from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
from sklearn.model_selection import KFold


from MEDC_EdLa import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC





def calculate_ACC(confusion_matrix, y_label):
	diag = 0
	for i in range(len(confusion_matrix)):
	  diag += confusion_matrix[i][i]
	ACC = diag/len(y_label)
	return ACC


mean_ACC = []
def get_ACC(X, y_label, names, splits = 10, kNeighbors = 5, Gamma = 0.1, c = 10):
	kf = KFold(n_splits = splits)

	for n in range(len(X)):
		Accuracies = [[], [], []]
		for train_index, test_index in kf.split(X[n]):
			X_train, X_test = X[n][train_index], X[n][test_index]
			y_train, y_test = y_label[n][train_index], y_label[n][test_index]


			##############
			vectores_promedio = Vectores_Promedio(X_train, y_train)
			distance_matrix_MEDC = distancia_euclidiana(X_train, vectores_promedio)
			predicted_labels_MEDC = asignar_membresia(distance_matrix_MEDC)

			A = predict_distance(X_test, vectores_promedio)
			confusion_matrix_MEDC = skl.confusion_matrix(y_test, A)
			Accuracies[0].append(calculate_ACC(confusion_matrix_MEDC, A))


			##############
			kNN = KNeighborsClassifier(n_neighbors = kNeighbors)
			kNN.fit(X_train, y_train)
			predicted_labels_kNN = kNN.predict(X_test)

			confusion_matrix_kNN = skl.confusion_matrix(y_test, predicted_labels_kNN)

			Accuracies[1].append(calculate_ACC(confusion_matrix_kNN, predicted_labels_kNN))
		

			##############
			svc = SVC(gamma = Gamma, C = c)
			svc.fit(X_train, y_train)

			predicted_labels_svc = svc.predict(X_test)
			confusion_matrix_svc = skl.confusion_matrix(y_test, predicted_labels_svc)
			Accuracies[2].append(calculate_ACC(confusion_matrix_svc, predicted_labels_svc))



		mean_ACC.append([np.mean(Accuracies[0]), np.mean(Accuracies[1]), np.mean(Accuracies[2])])
	
	print('\n\n\n\n')

	print(np.array(names))
	print(np.array(mean_ACC))

	print('\n\n\n\n')