import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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



def plotter_function(X, y_label, datasets_names):
	plt.figure(figsize=(5, 5))
	plt.subplot(341)
	plt.scatter(X[0][:,0], X[0][:,1],  c=y_label[0], s=2, cmap=colormap)
	plt.title(datasets_names[0])

	plt.subplot(345)
	plt.scatter(X[1][:,0], X[1][:,1],  c=y_label[1], s=2, cmap=colormap)
	plt.title(datasets_names[1])

	plt.subplot(349)
	plt.scatter(X[2][:,0], X[2][:,1],  c=y_label[2], s=2, cmap=colormap)
	plt.title(datasets_names[2])

	plt.show()

datasets_names = ['dataset_classifiers1.csv', 'dataset_classifiers2.csv', 'dataset_classifiers3.csv']
colormap = plt.cm.RdYlBu
names = ['MEDC', 'k-NN', 'SVC']


datasets = generate_datasets(datasets_names)
X, y_label = clean_datasets(datasets)
#print(y_label[2])
plotter_function(X,y_label, names)