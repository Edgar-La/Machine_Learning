import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs, make_moons

#Making instances using -make_blobs()- method
def crear_puntos(n_MUESTRAS, k_CENTROS, std_dev, rnd_state):
	training_data , labels = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std = std_dev, random_state= rnd_state)
	return training_data, labels

def matrix_distances(X, Z):
  matrix = []
  for z in range(len(Z)):
    aux = []
    for n in range(len(X)):
      aux.append(np.linalg.norm(Z[z] - X[n]))
    matrix.append(aux)

  return matrix

def index_minimun_distances_matrix(k, matrix):
  matrix_min_index = []
  for n in range(len(matrix)):
    matrix_min_index.append(np.argpartition(matrix[n], k)[:k])  #https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
  return matrix_min_index

def clasifier_matrix(matrix_min_index, labels):
  matrix_labels = []
  for n in range(len(matrix_min_index)):
    aux = []
    for i in range(len(matrix_min_index[0])):
      aux.append(labels[matrix_min_index[n][i]])
    matrix_labels.append(aux)

  return matrix_labels

def get_mode(matrix_labels):
  predicted_labels = []
  for n in range(len(matrix_labels)):
    predicted_labels.append(np.bincount(matrix_labels[n]).argmax()) #https://www.geeksforgeeks.org/find-the-most-frequent-value-in-a-numpy-array/
  return np.array(predicted_labels)

def plotter_function(X_, Z_ ,label_ ,label2_ ,color_map_, color_, k_):
  plt.figure(figsize=(9, 9))

  plt.subplot(221)
  plt.scatter(X_[:,0], X_[:,1], c=label_, s=50, cmap=color_map_)
  plt.title('N-Instances')

  plt.subplot(222)
  plt.scatter(Z_[:,0], Z_[:,1], c=color_, s=100, alpha = .5)
  plt.title('test-Z')

  plt.subplot(223)
  plt.scatter(X_[:,0], X_[:,1], c=label_, s=50, cmap=color_map_)
  plt.scatter(Z_[:,0], Z_[:,1], c=color_, s=100, alpha = .5)
  plt.title('Without classifier')

  plt.subplot(224)
  plt.scatter(X_[:,0], X_[:,1], c=label_, s=50, cmap=color_map_)
  plt.scatter(Z_[:,0], Z_[:,1], c=label2_, s=50,  cmap=color_map_)
  plt.title('Using ' + str(k_)+ '-NN')

  plt.show()




X, labels = crear_puntos(200, 3, 0.5, 2)
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='spring')

#Create test-Z values
Z, labels_2 = crear_puntos(10, 3, 0.99, 2)
plt.scatter(Z[:,0], Z[:,1], c='black', s=100, alpha = .5)
#print(Z)

#10x200 Matrix: Distance test-Z and N-instances of X
matrix = matrix_distances(X, Z)
print(pd.DataFrame(matrix))

#https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
k_=5
matrix_min_index = index_minimun_distances_matrix(k_, matrix)
matrix_min_index

matrix_labels = clasifier_matrix(matrix_min_index, labels)
matrix_labels

predicted_labels = get_mode(matrix_labels)
print(predicted_labels)

print(labels_2)
print(predicted_labels)

plotter_function(X, Z, labels, labels_2, 'spring', 'black', k_)