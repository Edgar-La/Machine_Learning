import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Open a csv file with a random matrix
def open_file(name_file):
  return  pd.read_csv(name_file)

#Convert pandas datframe to numpy matrix
def dataframe_to_matrix(df):
  return np.array(df)

def generate_datasets(datasets):
	df_1 = open_file(datasets[0])
	df_2 = open_file(datasets[1])
	df_3 = open_file(datasets[2])

	return [dataframe_to_matrix(df_1), dataframe_to_matrix(df_2), dataframe_to_matrix(df_3)]

def plotter_function(matrix_1, matrix_2, matrix_3, datasets_names):
  plt.figure(figsize=(5, 5))

  plt.subplot(341)
  plt.scatter(matrix_1[:,1], matrix_1[:,2],  c=matrix_1[:,3], s=2, cmap=colormap)
  plt.title(datasets_names[0])

  plt.subplot(345)
  plt.scatter(matrix_2[:,1], matrix_2[:,2],  c=matrix_2[:,3], s=2, cmap=colormap)
  plt.title(datasets_names[1])

  plt.subplot(349)
  plt.scatter(matrix_3[:,0], matrix_3[:,1],  c=matrix_3[:,2], s=2, cmap=colormap)
  plt.title(datasets_names[2])

  plt.show()

datasets_names = ['dataset_classifiers1.csv', 'dataset_classifiers2.csv', 'dataset_classifiers3.csv']
colormap = plt.cm.RdYlBu
names = ['MEDC', 'k-NN', 'SVC']


datasets = generate_datasets(datasets_names)
plotter_function(datasets[0], datasets[1], datasets[2], names)