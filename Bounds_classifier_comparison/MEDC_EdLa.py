import numpy as np
from statistics import mean

#Vectores promedio para cada una de las 3 clases
def Vectores_Promedio(training_data, labels):
  #Hagdo una copia del arreglo de training_data
  training_data_copy = training_data.copy(); training_data_copy = training_data_copy.tolist()

  #Clasifico los puntos(200) en una lista de 3 elementos(3 clases)
  arreglo_aux = [[], []]
  for n in range(len(training_data_copy)):
    if labels[n] == 0:
      arreglo_aux[0].append(training_data_copy[n])
    elif labels[n] == 1:
      arreglo_aux[1].append(training_data_copy[n])

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

def asignar_membresia(distance_matrix):
  #distance_matrix[n] = matrix[n].copy()
  #distance_matrix[n] = distance_matrix[n].tolist()
  predicted_labels = []
  for n in range(len(distance_matrix)):
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
  return predicted_labels#, extended_distance_matrix

def predict_distance(data, vectores_promedio):
	distance_matrix = distancia_euclidiana(data, vectores_promedio)
	predicted_labels = asignar_membresia(distance_matrix)
	return predicted_labels


def run_MEDC(X, y_label, xx, yy):
  vectores_promedio = Vectores_Promedio(X, y_label)
  distance_matrix = distancia_euclidiana(X, vectores_promedio)
  predicted_labels = asignar_membresia(distance_matrix)

  A = predict_distance(np.c_[xx.ravel(), yy.ravel()], vectores_promedio)
  Z = A.reshape(xx.shape)


  return Z
