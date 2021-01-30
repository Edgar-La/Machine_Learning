#Edgar Lara Arellano
#Algoritmo de Clustering k-Means

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from random import randint
from math import dist
import os
#os.system('clear')


#Hay k=3 grupos, y n=200 puntos
#Se hacen globales las variables para poderlas usar en las siguientes funciones
global k_CENTROS; k_CENTROS = 3;
global n_MUESTRAS; n_MUESTRAS = 300;


#Funcion que llama al metodo -make_blobs()- visto en clase
def crear_puntos():
	X, y_true = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std=.90, random_state=3)
	return X


#Centroides seleccionados aleatoriamente a partir de las entradas
def indicar_centroides_iniciales(X):
	centros_random = []
	for n in range(k_CENTROS): #k iteraciones
		pos_random = randint(0, n_MUESTRAS) #entero aleatorio de 0 a n
		centros_random.append(X[pos_random]) #guardo en un vector la posicion aleatoria
		#print(centros_random[n])
	centros_random = np.array(centros_random)
	#print(type(centros_random))
	'''centros_random = np.array([X[197], X[198], X[199]])
	print(centros_random); print(type(centros_random))'''
	return centros_random


def asignar_membresia(X, centros_random):
	gamma = []
	for i in range(n_MUESTRAS):
		dista_i_punto = []; renglon_membresia = []
		for k in range(k_CENTROS):		#Calcula las distancias del punto i con los k centros
			dista_i_punto.append(dist(X[i], centros_random[k]))
		
		for k in range(k_CENTROS) :		#Asigna un 1 a la distancia minima, y 0 a las demás
			if dista_i_punto[k] == min(dista_i_punto):
				renglon_membresia.append(1)
			else:
				renglon_membresia.append(0)
		
		gamma.append(renglon_membresia)	#Construyo la matriz gamma agregando los renglones membresia
	gamma = np.array(gamma)
	#print(gamma)
	return gamma

def recalcular_centroides(gamma, X):
	cant_puntos_cluster = []; centros_recalculados = []
	for k in range(k_CENTROS):	#Cuenta los puntos en un cluster
		cant_puntos_cluster.append(sum(gamma[:,k]))
	print(cant_puntos_cluster)

	for k in range(k_CENTROS):
		suma = 0
		for i in range(n_MUESTRAS):
			suma += gamma[i][k]*X[i]	#Recalcula el centroide haciendo un promedio
		centros_recalculados.append(suma/cant_puntos_cluster[k])
	centros_recalculados = np.array(centros_recalculados)
	print(centros_recalculados)
	return centros_recalculados

#Verifica en que posicion de gamma esta el 1, y acorde a eso designa el cluster de 0-k
def colores_cluster(gamma):
	clusters = []
	for n in range(len(gamma)):
		cluster = 1
		for k in range(k_CENTROS):
			if gamma[n][k] == 1:
				clusters.append(cluster)
			else:
				cluster += 1
	return clusters


#Funcion para ajustes del gráfico
def graficar_puntos_generados(X, centros_random, centros_recalculados, clusters):
	f = plt.figure(figsize=(15,4.5))
	puntos_originales = f.add_subplot(121)
	puntos_con_centroides = f.add_subplot(122)

	puntos_originales.scatter(X[:,0], X[:,1], s=50, label = 'Puntos aleatorios')
	puntos_originales.scatter(centros_random[:,0], centros_random[:,1], c='black', s=200, alpha = .5, label = 'Centros aleatorios')
	puntos_originales.legend()

	puntos_con_centroides.scatter(X[:,0], X[:,1], c=clusters, s=50, cmap='spring', label = 'Puntos asociados a un cluster')
	puntos_con_centroides.scatter(centros_recalculados[:,0], centros_recalculados[:,1], c='black', s=200, alpha = .5, label = 'Centroides recalculados')
	puntos_con_centroides.legend()

	plt.show()	






#Sección donde se llama a las funciones
X = crear_puntos()
centros_random = indicar_centroides_iniciales(X)
gamma = asignar_membresia(X, centros_random)
centros_recalculados = recalcular_centroides(gamma, X)
clusters = colores_cluster(gamma)
graficar_puntos_generados(X, centros_random, centros_recalculados, clusters)