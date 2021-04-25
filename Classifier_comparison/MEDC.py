#Seccion para importar modulos
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

#Funcion que llama al metodo -make_blobs()-
def crear_puntos(n_MUESTRAS, k_CENTROS, std_dev, rnd_state):
	training_data , labels = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std = std_dev, random_state= rnd_state)
	return training_data, labels

#Vectores promedio para cada una de las 3 clases
def Vectores_Promedio(training_data, labels):
  #Hagdo una copia del arreglo de training_data
  training_data_copy = training_data.copy(); training_data_copy = training_data_copy.tolist()

  #Clasifico los puntos(200) en una lista de 3 elementos(3 clases)
  arreglo_aux = [[], [], []]
  for n in range(len(training_data_copy)):
    if labels[n] == 0:
      arreglo_aux[0].append(training_data_copy[n])
    elif labels[n] == 1:
      arreglo_aux[1].append(training_data_copy[n])
    elif labels[n] == 2:
      arreglo_aux[2].append(training_data_copy[n])

  #Uso numpy por facilidad
  for n in range(len(arreglo_aux)):
    arreglo_aux[n] = np.array(arreglo_aux[n])
  
  #Calculo el promedio de c/una de las clases
  vectores_promedio = []
  for n in range(len(arreglo_aux)):
    vectores_promedio.append(np.mean(arreglo_aux[n],  axis=0))
  vectores_promedio = np.array(vectores_promedio)
  #print(vectores_promedio)

  return vectores_promedio

def distancia_euclidiana(test_data, vectores_promedio):
  distance_matrix = []
  for n in range(len(test_data)):
    aux = []
    for i in range(len(vectores_promedio)):
      aux.append(np.linalg.norm(test_data[n] - vectores_promedio[i]))
    distance_matrix.append(aux)
  distance_matrix = np.array(distance_matrix)
  #print(distance_matrix)
  return distance_matrix

def asignar_membresia(matrix, test_data):
  #distance_matrix[n] = matrix[n].copy()
  #distance_matrix[n] = distance_matrix[n].tolist()
  predicted_labels = []
  for n in range(len(matrix)):
    minimum = min(distance_matrix[n])
    contador_membresia = 0
    for i in range(len(distance_matrix[n])):
      if minimum == distance_matrix[n][i]:
        predicted_labels.append(contador_membresia)
      else:
        contador_membresia += 1

  predicted_labels = np.array(predicted_labels)
  #print(predicted_labels)

  aux = []
  for n in range(len(predicted_labels)):
    aux.append([predicted_labels[n]])
  aux = np.array(aux)
  #distance_matrix = np.append(distance_matrix, predicted_labels, axis=1)
  extended_distance_matrix = np.append(distance_matrix, aux, axis=1)

  #print(new)
  return predicted_labels, extended_distance_matrix

def comparar_labels(labels, predicted_labels):
  aciertos = 0; errores = 0;
  for n in range(len(predicted_labels)):
    if labels[n] == predicted_labels[n]:
      aciertos += 1
    else:
      errores+=1
  print('labesl = ', labels)
  print('predicted_labels = ', predicted_labels)
  print('aciertos: ', aciertos)
  print('errores: ', errores)
  return aciertos, errores

#Paso 1
training_data, labels = crear_puntos(200, 3, 0.5, 2)
plt.scatter(training_data[:,0], training_data[:,1], c=labels, s=50, cmap='spring')
plt.show()

#Paso 2
test_data, y = crear_puntos(10, 3, 0.5, 2)
plt.scatter(test_data[:,0], test_data[:,1], c='black', s=100, alpha = .5)
plt.show()

#Paso 3
vectores_promedio = Vectores_Promedio(training_data, labels)
plt.scatter(training_data[:,0], training_data[:,1], c=labels, s=50, cmap='spring')
plt.scatter(vectores_promedio[:,0], vectores_promedio[:,1], c='black', s=250, alpha = .45)
plt.show()

#Paso 4
distance_matrix = distancia_euclidiana(test_data, vectores_promedio)
print(distance_matrix)

#Paso 5
predicted_labels, extended_distance_matrix = asignar_membresia(distance_matrix, test_data)
print('predicted_labels = ', predicted_labels)
print('Extended matrix = \n')
print(extended_distance_matrix)

#Paso 6
aciertos, errores = comparar_labels(y, predicted_labels)

#Paso extra
plt.scatter(training_data[:,0], training_data[:,1], c=labels, s=50, cmap='spring', label = 'training_data')
plt.scatter(vectores_promedio[:,0], vectores_promedio[:,1], c='black', s=250, alpha = .45, label = 'vectores_promedio')
plt.scatter(test_data[:,0], test_data[:,1], c='green', s=100, alpha = .5, label = 'test_data')
plt.legend()
plt.title('Puntos graficados pero aun SIN clasificar')
plt.show()
