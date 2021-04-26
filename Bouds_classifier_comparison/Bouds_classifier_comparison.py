import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from k_NN_EdLa import run_kNN
from SVC_EdLa import run_SVC
from MEDC_EdLa import run_MEDC



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



def plotter_function(X, y_label, datasets_names, xx, yy, Z):
	plt.figure(figsize=(5, 5))

	plt.subplot(3,4,1)
	plt.scatter(X[0][:,0], X[0][:,1],  c=y_label[0], s=2, cmap=colormap)
	plt.title(datasets_names[0])
	
	
	plt.subplot(3,4,2)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_MEDC[0], yy_MEDC[0], Z_MEDC[0], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[1])

	
	plt.subplot(3,4,3)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[0], yy[0], Z[0], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[2])

	plt.subplot(3,4,4)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_SVC[0], yy_SVC[0], Z_SVC[0], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[3])

	plt.subplot(3,4,5)
	plt.scatter(X[1][:,0], X[1][:,1],  c=y_label[1], s=2, cmap=colormap)

	plt.subplot(3,4,6)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_MEDC[1], yy_MEDC[1], Z_MEDC[1], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,7)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[1], yy[1], Z[1], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')

	plt.subplot(3,4,8)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_SVC[1], yy_SVC[1], Z_SVC[1], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,9)
	plt.scatter(X[2][:,0], X[2][:,1],  c=y_label[2], s=2, cmap=colormap)

	plt.subplot(3,4,10)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_MEDC[2], yy_MEDC[2], Z_MEDC[2], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,11)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[2], yy[2], Z[2], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')

	plt.subplot(3,4,12)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx_SVC[2], yy_SVC[2], Z_SVC[2], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.show()

datasets_names = ['dataset_classifiers1.csv', 'dataset_classifiers2.csv', 'dataset_classifiers3.csv']
colormap = plt.cm.RdYlBu
names = ['Original Data','MEDC', 'k-NN', 'SVC']
h = .5	################################################

datasets = generate_datasets(datasets_names)
X, y_label = clean_datasets(datasets)


xx = []; yy = []; Z = []
for n in range(len(datasets)):
	xx_, yy_, Z_ = run_kNN(X[n], y_label[n], h)
	xx.append(xx_); yy.append(yy_); Z.append(Z_)

xx_SVC = []; yy_SVC = []; Z_SVC = []
for n in range(len(datasets)):
	xx_s, yy_s, Z_s = run_SVC(X[n], y_label[n], h)
	xx_SVC.append(xx_s); yy_SVC.append(yy_s); Z_SVC.append(Z_s)

xx_MEDC = []; yy_MEDC = []; Z_MEDC = []
for n in range(len(datasets)):
	xx_MEDC_, yy_MEDC_, Z_MEDC_ = run_MEDC(X[n], y_label[n], h)
	xx_MEDC.append(xx_MEDC_); yy_MEDC.append(yy_MEDC_); Z_MEDC.append(xx_MEDC_);

plotter_function(X, y_label, names, xx, yy, Z)