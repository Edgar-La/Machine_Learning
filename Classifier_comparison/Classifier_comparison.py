import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kMeans import *


#Hay --k-- centros y --n-- puntos
#Se hacen globales las variables para poderlas usar en cualquier funcion
global k_CENTROS; global n_MUESTRAS; 
k_CENTROS = 3; n_MUESTRAS = 299;
#k_CENTROS = 3; n_MUESTRAS = 200;


#Funcion que llama al metodo -make_blobs()- visto en clase
def crear_puntos():
	X, y_true = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std=.90, random_state=3)
	#X, y_true = make_blobs(n_samples=n_MUESTRAS, centers=k_CENTROS, cluster_std=1, random_state=1)
	return X

#Funcion para ajustes del gráfico
def graficar_puntos_generados(X, centros_random, centros_recalculados, clusters, labels_cluster):
	global cont_iteraciones
	f = plt.figure(figsize=(15,4.5))
	puntos_originales = f.add_subplot(121)
	puntos_con_centroides = f.add_subplot(122)

	#Grafica de la izquuierda: ----Random----
	puntos_originales.scatter(X[:,0], X[:,1], s=50, label = 'Puntos aleatorios')
	puntos_originales.scatter(centros_random[:,0], centros_random[:,1], c='black', s=200, alpha = .5, label = 'Centros aleatorios')
	puntos_originales.legend()

	#Grafica de la derecha: ----Recalculado----
	#Para poner 2 leyendas (una de colores) tuve que buscar documentacion: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html
	f_1 = puntos_con_centroides.scatter(X[:,0], X[:,1], c=clusters, s=50, cmap='spring')
	f_2 = puntos_con_centroides.scatter(centros_recalculados[:,0], centros_recalculados[:,1], c='black', s=200, alpha = .5, label = 'Centroides recalculados')
	l_1 = puntos_con_centroides.legend(handles = f_1.legend_elements()[0], labels=labels_cluster, title = 'Clusters', loc='lower right')
	l_2 = puntos_con_centroides.legend()
	puntos_con_centroides.add_artist(l_1)
	puntos_con_centroides.add_artist(l_2)
	puntos_con_centroides.title.set_text('Iteraciones del algoritmo: ' + str(cont_iteraciones))
	plt.show()	

X = crear_puntos()


centros_random, centros_recalculados, clusters, labels_cluster = run_kMeans(X, k_CENTROS, n_MUESTRAS)

#graficar_puntos_generados(X, centros_random, centros_recalculados, clusters, labels_cluster)
plt.scatter(X[:,0], X[:,1], c=clusters, s=50, cmap='spring')
plt.scatter(centros_recalculados[:,0], centros_recalculados[:,1], c='black', s=200, alpha = .5, label = 'Centroides recalculados')
plt.show()
input()
