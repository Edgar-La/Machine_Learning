import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plotter_function(X, y_label, datasets_names, xx, yy, Z):
	colormap = plt.cm.RdYlBu
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  

	plt.figure(figsize=(5, 5))
	k = 0
	i =-1
	first_iter = True
	for n in range(len(X)):
		k+=1
		plt.subplot(3,4,k)
		plt.scatter(X[n][:,0], X[n][:,1],  c=y_label[n], s=2, cmap=colormap)
		if first_iter: plt.title(datasets_names[0])

		k+=1; i+=1
		plt.subplot(3,4,k)
		plt.pcolormesh(xx[i], yy[i], Z[i], cmap=cmap_light)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cmap_bold,edgecolor='k')	
		if first_iter: plt.title(datasets_names[1])

		k+=1; i+=1
		plt.subplot(3,4,k) 
		plt.pcolormesh(xx[i], yy[i], Z[i], cmap=cmap_light)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cmap_bold,edgecolor='k')	
		if first_iter: plt.title(datasets_names[2])

		k+=1; i+=1
		plt.subplot(3,4,k) 
		plt.pcolormesh(xx[i], yy[i], Z[i], cmap=cmap_light)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cmap_bold,edgecolor='k')	
		if first_iter: plt.title(datasets_names[3])

		first_iter = False

	plt.show()