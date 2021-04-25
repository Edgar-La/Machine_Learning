#import matplotlib.pyplot as plt
#from sklearn.datasets import make_blobs
import numpy as np
from random import randint
from math import dist


#Centroides seleccionados aleatoriamente a partir de las entradas
def indicar_centroides_iniciales(X, k_CENTROS, n_MUESTRAS):
	centros_random = []
	for n in range(k_CENTROS): #k iteraciones
		pos_random = randint(0, n_MUESTRAS) #entero aleatorio de 0 a n
		centros_random.append(X[pos_random]) #guardo en un vector la posicion aleatoria
	centros_random = np.array(centros_random)
	return centros_random


def asignar_membresia(X, centros_random, k_CENTROS, n_MUESTRAS):
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

def recalcular_centroides(gamma, X, k_CENTROS, n_MUESTRAS):
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



def lograr_convergencia(X, centros_random, gamma, centros_recalculados_, k_CENTROS, n_MUESTRAS):
	global cont_iteraciones
	vector_aux = []; cont_iteraciones = 1; check = 'Not Yet'; centros_recalculados = centros_recalculados_.copy()
	while check != 'convergencia':
		cont_iteraciones += 1
		#Repito las funciones de membresia y de calculo
		gamma = asignar_membresia(X, centros_recalculados, k_CENTROS, n_MUESTRAS)
		centros_recalculados, cant_puntos_cluster = recalcular_centroides(gamma, X, k_CENTROS, n_MUESTRAS)
		#Pregunto si alcanzó convergencia
		if (np.array_equal(vector_aux, centros_recalculados) == True):
			#Si converge termina el ciclo
			check = 'convergencia'
			print('\nIteraciones del algoritmo = ', cont_iteraciones)
			print('\nCentros iniciales random = \n', centros_random)
			print('\nCentros recalculados = \n', centros_recalculados)
			print('\nLos puntos en c/centroide = \n', cant_puntos_cluster)
		#Si aún no converge guarda los centroides(anteriores) en una var_aux y se repite
		else:
			vector_aux = centros_recalculados.copy()
	return centros_recalculados, gamma, cant_puntos_cluster

#Verifica en que posicion de gamma esta el 1, y acorde a eso designa el cluster de 0-k
def colores_cluster(gamma, k_CENTROS):
	clusters = []
	for n in range(len(gamma)):
		cluster = 1
		for k in range(k_CENTROS):
			if gamma[n][k] == 1:
				clusters.append(cluster)
			else:
				cluster += 1
	return clusters


def etiquetas_cluster_plot(cant_puntos_cluster, k_CENTROS):
	etiquetas_cluster = []
	for n in range(k_CENTROS):
		etiquetas_cluster.append('Cluster ' + str(n+1) + ': ' + str(cant_puntos_cluster[n]) + ' puntos')
	print(etiquetas_cluster)
	return etiquetas_cluster


def run_kMeans(X, k_CENTROS, n_MUESTRAS):
	centros_random = indicar_centroides_iniciales(X, k_CENTROS, n_MUESTRAS)
	gamma_ = asignar_membresia(X, centros_random, k_CENTROS, n_MUESTRAS)
	centros_recalculados_, cant_puntos_cluster = recalcular_centroides(gamma_, X, k_CENTROS, n_MUESTRAS)
	#Esta funcion es importante va a repetir las membresias y calculos hasta lograr una convergencia
	centros_recalculados, gamma, cant_puntos_cluster =  lograr_convergencia(X, centros_random, gamma_, centros_recalculados_, k_CENTROS, n_MUESTRAS)
	clusters = colores_cluster(gamma, k_CENTROS)
	labels_cluster = etiquetas_cluster_plot(cant_puntos_cluster,  k_CENTROS)
	#graficar_puntos_generados(X, centros_random, centros_recalculados, clusters, labels_cluster)
	#input()
	return centros_random, centros_recalculados, clusters, labels_cluster
