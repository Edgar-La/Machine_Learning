#import os; os.system('clear')		#Clean terminal

from Read_datasets_EdLa import *	#Import self scripts
from MEDC_EdLa import run_MEDC
from k_NN_EdLa import run_kNN
from SVC_EdLa import run_SVC
from Accuracy_kFold_EdLa import *
from Plotter_EdLa import *
from Perceptron_module import run_Perceptron
from Percep_skl_module import run_Percep_skl

################ The user can modify this values ################
#--------------------------------------------------------------------------
datasets_names = ['Data/dataset_classifiers1.csv',		#Datasets names
					'Data/dataset_classifiers2.csv',
					'Data/dataset_classifiers3.csv']

h = .09					#how accurate will the mesh be 
k_Neighbors = 5			#Neighbors for the k-NN method
Gamma_ = 0.1			#Value for SVC
c_ = 10					#Value for SVC
folds = 10				#Folder number in cross validation
N_Epochs = 1			#Value for perceptron
Learning_step = .5	#Learning ratio
#--------------------------------------------------------------------------

names = ['MEDC', 'k-NN', 'SVC', 'Perceptron', 'Percep_skl']						#Classifiers names

#This method read all datasets
X, y_label = read_datasets(datasets_names)

#Loop that run the methods for every dataset
xx = []; yy = []; Z = []
for n in range(len(X)):
	x_min, x_max = X[n][:, 0].min() - 1, X[n][:, 0].max() + 1
	y_min, y_max = X[n][:, 1].min() - 1, X[n][:, 1].max() + 1
	xx_, yy_ = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	xx.append(xx_); yy.append(yy_);

	Z_MEDC = run_MEDC(X[n], y_label[n], xx[n], yy[n])
	Z.append(Z_MEDC);

	Z_kNN = run_kNN(X[n], y_label[n], xx[n], yy[n], kNeighbors = k_Neighbors)
	Z.append(Z_kNN)

	Z_SVC = run_SVC(X[n], y_label[n], xx[n], yy[n], Gamma = Gamma_, c = c_)
	Z.append(Z_SVC)

	Z_Perceptron = run_Perceptron(X[n], y_label[n], xx[n], yy[n], Epochs=N_Epochs, L_step = Learning_step)
	Z.append(Z_Perceptron)

	Z_Percep_skl = run_Percep_skl(X[n], y_label[n], xx[n], yy[n], L_step = Learning_step)
	Z.append(Z_Percep_skl)

#This method obtains the mean accuracy for every datasets and every method
#but using --CROSS VALIDATION--
#get_ACC(X, y_label, names, splits = folds, kNeighbors = k_Neighbors, Gamma = Gamma_, c = c_, Epochs=N_Epochs, L_step = Learning_step)

#This method make the plot
plotter_function(X, y_label, names, xx, yy, Z)