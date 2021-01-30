#The following code was seen in the course
#I am not the autor of this one
#The other code (where I dont use KMeans) in this folder was coded by me

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


#Call the method -make_blobs()- to create points
#k=3 clusters and n=200 points
def crear_puntos():
	X_, y_true = make_blobs(n_samples=200, centers=3, cluster_std=1, random_state=1)
	return X_

#Training function
def entrenar_con_kmeans(X_):
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(X_)
	y_kmeans = kmeans.predict(X_)
	centers = kmeans.cluster_centers_
	return y_kmeans, centers

#Plot settings
def graficar_puntos_generados(X_, y_kmeans, centers):
	plt.scatter(X_[:,0], X_[:,1], c=y_kmeans, s=50, cmap='spring')
	plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha = .5)
	plt.show()	


#Section where I call the functions
X = crear_puntos()
y_kmeans, centers = entrenar_con_kmeans(X)
graficar_puntos_generados(X, y_kmeans, centers)
