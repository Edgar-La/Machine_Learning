import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plotter_function(X, y_label, datasets_names, xx, yy, Z):
	colormap = plt.cm.RdYlBu

	plt.figure(figsize=(5, 5))

	plt.subplot(3,4,1)
	plt.scatter(X[0][:,0], X[0][:,1],  c=y_label[0], s=2, cmap=colormap)
	plt.title(datasets_names[0])
	
	
	plt.subplot(3,4,2)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[0], yy[0], Z[0], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[1])

	
	plt.subplot(3,4,3)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[1], yy[1], Z[1], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[2])

	plt.subplot(3,4,4)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[2], yy[2], Z[2], cmap=cmap_light)
	plt.scatter(X[0][:, 0], X[0][:, 1], c=y_label[0], s=7, cmap=cmap_bold,edgecolor='k')	
	plt.title(datasets_names[3])

	plt.subplot(3,4,5)
	plt.scatter(X[1][:,0], X[1][:,1],  c=y_label[1], s=2, cmap=colormap)

	plt.subplot(3,4,6)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[3], yy[3], Z[3], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,7)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[4], yy[4], Z[4], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')

	plt.subplot(3,4,8)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[5], yy[5], Z[5], cmap=cmap_light)
	plt.scatter(X[1][:, 0], X[1][:, 1], c=y_label[1], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,9)
	plt.scatter(X[2][:,0], X[2][:,1],  c=y_label[2], s=2, cmap=colormap)

	plt.subplot(3,4,10)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[6], yy[6], Z[6], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.subplot(3,4,11)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[7], yy[7], Z[7], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')

	plt.subplot(3,4,12)
	cmap_light = ListedColormap(['#FFAAAA', '#ffcc99'])
	cmap_bold = ListedColormap(['#FF0000', '#ff9933'])  
	plt.pcolormesh(xx[8], yy[8], Z[8], cmap=cmap_light)
	plt.scatter(X[2][:, 0], X[2][:, 1], c=y_label[2], s=7, cmap=cmap_bold,edgecolor='k')	

	plt.show()