import pandas as pd
import numpy as np

def generate_datasets(datasets_names):
	dataset_1 = np.array(pd.read_csv(datasets_names[0], index_col = 0))
	dataset_2 = np.array(pd.read_csv(datasets_names[1], index_col = 0))
	#dataset_3 = np.array(pd.read_csv(datasets_names[2]))
	dataset_3 = np.array(pd.read_csv(datasets_names[2], index_col = 0))
	#dataset_4 = np.array(pd.read_csv(datasets_names[3], index_col = 0))
	#return [dataset_1, dataset_2, dataset_3, dataset_4]
	return [dataset_1, dataset_2, dataset_3]

def clean_datasets(datasets):
	#X = [[],[],[],[]]
	X = []
	#for m in range(len())
	for k in range(len(datasets)):
		X.append([])
		for n in range(len(datasets[k])):
			X[k].append([datasets[k][n][0],datasets[k][n][1]])
		X[k] = np.array(X[k])
	#y_label = [datasets[0][:,2], datasets[1][:,2], datasets[2][:,2], datasets[3][:,2]]
	y_label = [datasets[0][:,2], datasets[1][:,2], datasets[2][:,2]]
	return X, y_label





def read_datasets(datasets_names):
	datasets = generate_datasets(datasets_names)
	X, y_label = clean_datasets(datasets)
	return X, y_label