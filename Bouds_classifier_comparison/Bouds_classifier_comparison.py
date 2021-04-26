import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from k_NN_EdLa import run_kNN
from SVC_EdLa import run_SVC
from MEDC_EdLa import run_MEDC
from Plotter_EdLa import *


#Convert pandas datframe to numpy matrix
def dataframe_to_matrix(df):
  return np.array(df)

def generate_datasets(datasets):
	dataset_1 = np.array(pd.read_csv(datasets[0], index_col = 0))
	dataset_2 = np.array(pd.read_csv(datasets[1], index_col = 0))
	dataset_3 = np.array(pd.read_csv(datasets[2]))
	return [dataset_1, dataset_2, dataset_3]

def clean_datasets(datasets):
	X = [[],[],[]]
	for k in range(len(datasets)):
		for n in range(len(datasets[k])):
			X[k].append([datasets[k][n][0],datasets[k][n][1]])
		X[k] = np.array(X[k])
	y_label = [datasets[0][:,2], datasets[1][:,2], datasets[2][:,2]]
	return X, y_label




datasets_names = ['dataset_classifiers1.csv', 'dataset_classifiers2.csv', 'dataset_classifiers3.csv']
#colormap = plt.cm.RdYlBu
names = ['Original Data','MEDC', 'k-NN', 'SVC']
h = .07	################################################

datasets = generate_datasets(datasets_names)
X, y_label = clean_datasets(datasets)


xx = []; yy = []; Z = []
for n in range(len(datasets)):
	xx_MEDC, yy_MEDC, Z_MEDC = run_MEDC(X[n], y_label[n], h)
	xx.append(xx_MEDC); yy.append(yy_MEDC); Z.append(xx_MEDC);

	xx_kNN, yy_kNN, Z_kNN = run_kNN(X[n], y_label[n], h)
	xx.append(xx_kNN); yy.append(yy_kNN); Z.append(Z_kNN)

	xx_SVC, yy_SVC, Z_SVC = run_SVC(X[n], y_label[n], h)
	xx.append(xx_SVC); yy.append(yy_SVC); Z.append(Z_SVC)


plotter_function(X, y_label, names, xx, yy, Z)