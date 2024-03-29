import numpy as np
import sklearn.metrics as skl
from statistics import mean
#Help from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
from sklearn.model_selection import KFold

from MEDC_EdLa import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from Perceptron_module import *
from Percep_skl_module import *
from FFNN_module import *

import plotly.figure_factory as go

#This method calulate the accuracy
def calculate_ACC(confusion_matrix, y_label):
	diag = 0
	for i in range(len(confusion_matrix)):
	  diag += confusion_matrix[i][i]
	ACC = diag/len(y_label)
	return ACC

#This method is call by the MAIN SCRIPT
#############################################################################
mean_ACC = []
def get_ACC(X, y_label, names, datasets_names, splits = 10, kNeighbors = 5, Gamma = 0.1, c = 10, Epochs=1, L_step = .005):
	kf = KFold(n_splits = splits)

	for n in range(len(X)):
		Accuracies = [[], [], [], [], [], []]
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

			##############
			train_x = pd.DataFrame({'x1':X_train[:,0],'x2':X_train[:,1]})
			train_y = pd.DataFrame({'':y_train})


			my_perceptron = Perceptron_(0.1,0.1)
			my_perceptron.fit(train_x, train_y, epochs=Epochs, step=L_step)

			#data = np.c_[xx.ravel(), yy.ravel()]
			test_x = pd.DataFrame({'x1':X_test[:,0],'x2':X_test[:,1]})

			predicted_labels_Perceptron = np.array(test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2), axis=1))
			confusion_matrix_Perceptron = skl.confusion_matrix(y_test, predicted_labels_Perceptron)
			Accuracies[3].append(calculate_ACC(confusion_matrix_Perceptron, predicted_labels_Perceptron))


			##############
			clf = Perceptron(tol=L_step, random_state=0)
			clf.fit(X_train, y_train)

			predicted_labels_Percep_skl = clf.predict(X_test)
			confusion_matrix_Percep_skl = skl.confusion_matrix(y_test, predicted_labels_Percep_skl)
			Accuracies[4].append(calculate_ACC(confusion_matrix_Percep_skl, predicted_labels_Percep_skl))

			##############
			train_x_FFNN, train_y_FFNN = convert_values(X_train, y_train)
			test_X_FFNN = pd.DataFrame(data=X_test, columns=["x1", "x2"])

			my_network = FFNN()
			if n==3: #Only Dataset 4 will run with 1000 epochs
				my_network.fit(train_x_FFNN, train_y_FFNN, epochs=1000, step=L_step)
			else:
				my_network.fit(train_x_FFNN, train_y_FFNN, epochs=Epochs, step=L_step)

			pred_y = test_X_FFNN.apply(my_network.forward, axis=1)
	
			predicted_labels_FFNN = get_binary_labels(pred_y)

			confusion_matrix_FFNN = skl.confusion_matrix(y_test, predicted_labels_FFNN)
			Accuracies[5].append(calculate_ACC(confusion_matrix_FFNN, predicted_labels_FFNN))


		mean_ACC.append([np.mean(Accuracies[0]), np.mean(Accuracies[1]), np.mean(Accuracies[2]), np.mean(Accuracies[3]), np.mean(Accuracies[4]), np.mean(Accuracies[5])])


	df = pd.DataFrame(np.array(mean_ACC))
	df.columns = names
	df.insert(0, 'datasets', ['Dataset1',		#Datasets names
					'Dataset2',
					'Dataset3',
					'Dataset4'])
	print(df)

	
	fig = go.create_table(df)
	#Probably need the orca package, try (linux) : conda install -c plotly plotly-orca
	#fig.write_image('FFNN.png', scale = 2, width=1000)
	fig.show()