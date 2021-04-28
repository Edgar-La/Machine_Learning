import os; os.system('clear')		#Clean terminal

from Read_datasets_EdLa import *	#Import self scripts
from MEDC_EdLa import run_MEDC
from k_NN_EdLa import run_kNN
from SVC_EdLa import run_SVC
from Accuracy_kFold_EdLa import *
from Plotter_EdLa import *



datasets_names = ['dataset_classifiers1.csv', 'dataset_classifiers2.csv', 'dataset_classifiers3.csv']
names = ['MEDC', 'k-NN', 'SVC']
h = .09

k_Neighbors = 5
Gamma_ = 0.1
c_ = 10


X, y_label = read_datasets(datasets_names)



xx = []; yy = []; Z = []
for n in range(len(X)):
	x_min, x_max = X[n][:, 0].min() - 1, X[n][:, 0].max() + 1
	y_min, y_max = X[n][:, 1].min() - 1, X[n][:, 1].max() + 1
	xx_, yy_ = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	xx.append(xx_); yy.append(yy_);

for n in range(len(X)):
	Z_MEDC = run_MEDC(X[n], y_label[n], xx[n], yy[n])
	Z.append(Z_MEDC);

	Z_kNN = run_kNN(X[n], y_label[n], xx[n], yy[n], kNeighbors = k_Neighbors)
	Z.append(Z_kNN)

	Z_SVC = run_SVC(X[n], y_label[n], xx[n], yy[n], Gamma = Gamma_, c = c_)
	Z.append(Z_SVC)


plotter_function(X, y_label, names, xx, yy, Z)

get_ACC(X, y_label, names, splits = 10, kNeighbors = k_Neighbors, Gamma = Gamma_, c = c_)