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
global k_CENTROS; global n_MUESTRAS; 
#k_CENTROS = 3; n_MUESTRAS = 300;
k_CENTROS = 3; n_MUESTRAS = 200;


#Funcion que llama al metodo -make_blobs()- visto en clase
def crear_puntos():
	#X, y_true = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std=.90, random_state=3)
	X, y_true = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std=1, random_state=1)
	return X


#Centroides seleccionados aleatoriamente a partir de las entradas
def indicar_centroides_iniciales(X):
	centros_random = []
	for n in range(k_CENTROS): #k iteraciones
		pos_random = randint(0, n_MUESTRAS) #entero aleatorio de 0 a n
		centros_random.append(X[pos_random]) #guardo en un vector la posicion aleatoria
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
	return gamma

def recalcular_centroides(gamma, X):
	cant_puntos_cluster = []; centros_recalculados = []
	for k in range(k_CENTROS):	#Cuenta los puntos en un cluster
		cant_puntos_cluster.append(sum(gamma[:,k]))

	for k in range(k_CENTROS):
		suma = 0
		for i in range(n_MUESTRAS):
			suma += gamma[i][k]*X[i]	#Recalcula el centroide haciendo un promedio
		centros_recalculados.append(suma/cant_puntos_cluster[k])
	centros_recalculados = np.array(centros_recalculados)
	return centros_recalculados, cant_puntos_cluster



def lograr_convergencia(X, centros_random, gamma, centros_recalculados_):
	vector_aux = []; contador = 1; check = 'Not Yet'; centros_recalculados = centros_recalculados_.copy()
	while check != 'convergencia':
		contador += 1
		#Repito las funciones de membresia y de calculo
		gamma = asignar_membresia(X, centros_recalculados)
		centros_recalculados, cant_puntos_cluster = recalcular_centroides(gamma, X)
		#Pregunto si alcanzó convergencia
		if (np.array_equal(vector_aux, centros_recalculados) == True):
			#Si converge termina el ciclo
			check = 'convergencia'
			print('\nIteraciones del algoritmo = ', contador)
			print('\nCentros iniciales random = \n', centros_random)
			print('\nCentros recalculados = \n', centros_recalculados)
			print('\nLos puntos en c/centroide = \n', cant_puntos_cluster)
		#Si aún no converge guarda los centroides(anteriores) en una var_aux y se repite
		else:
			vector_aux = centros_recalculados.copy()
	return centros_recalculados, gamma, cant_puntos_cluster

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


def etiquetas_cluster_plot(cant_puntos_cluster):
	etiquetas_cluster = []
	for n in range(k_CENTROS):
		etiquetas_cluster.append('Cluster ' + str(n+1) + ': ' + str(cant_puntos_cluster[n]) + ' puntos')
	print(etiquetas_cluster)
	return etiquetas_cluster

#Funcion para ajustes del gráfico
def graficar_puntos_generados(X, centros_random, centros_recalculados, clusters, labels_cluster):
	f = plt.figure(figsize=(15,4.5))
	puntos_originales = f.add_subplot(121)
	puntos_con_centroides = f.add_subplot(122)

	puntos_originales.scatter(X[:,0], X[:,1], s=50, label = 'Puntos aleatorios')
	puntos_originales.scatter(centros_random[:,0], centros_random[:,1], c='black', s=200, alpha = .5, label = 'Centros aleatorios')
	puntos_originales.legend()

	#Uso for para que agregue etiquetas por c/cluster
	f_1 = puntos_con_centroides.scatter(X[:,0], X[:,1], c=clusters, s=50, cmap='spring')
	l_1 = puntos_con_centroides.legend(handles=f_1.legend_elements()[0], labels=labels_cluster, loc ="upper left")
	f_2 = puntos_con_centroides.scatter(centros_recalculados[:,0], centros_recalculados[:,1], c='black', s=200, alpha = .5, label = 'Centroides recalculados')
	#l_2 = puntos_con_centroides.legend(handles=f_2.legend_elements()[0], labels = 'Centroides recalculados', loc = 'lower right')



	plt.show()	






#Sección donde se llama a las funciones
X = crear_puntos()
centros_random = indicar_centroides_iniciales(X)
gamma_ = asignar_membresia(X, centros_random)
centros_recalculados_, cant_puntos_cluster = recalcular_centroides(gamma_, X)
#Esta funcion es importante va a repetir las membresias y calculos hasta lograr una convergencia
centros_recalculados, gamma, cant_puntos_cluster =  lograr_convergencia(X, centros_random, gamma_, centros_recalculados_)
clusters = colores_cluster(gamma)

labels_cluster = etiquetas_cluster_plot(cant_puntos_cluster)
graficar_puntos_generados(X, centros_random, centros_recalculados, clusters, labels_cluster)
