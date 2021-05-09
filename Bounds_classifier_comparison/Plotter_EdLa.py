import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


#This method is call by the MAIN SCRIPT
#############################################################################
def plotter_function(X, y_label, datasets_names, xx, yy, Z, Gamma= 0.1, c=10):
	colormap = plt.cm.RdYlBu 
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
	plt.figure(figsize=(20, 10))
	k = 0
	i =-1
	datasets_names.insert(0, 'Original Data')
	first_iter = True
	for n in range(len(X)):
		k+=1
		plt.subplot(3,6,k)
		plt.scatter(X[n][:,0], X[n][:,1],  c=y_label[n], s=2, cmap=cm_bright)
		if first_iter: plt.title(datasets_names[0])

		k+=1; i+=1
		plt.subplot(3,6,k)
		plt.pcolormesh(xx[n], yy[n], Z[i], cmap=colormap)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cm_bright,edgecolor='k')	
		if first_iter: plt.title(datasets_names[1])

		k+=1; i+=1
		plt.subplot(3,6,k) 
		plt.pcolormesh(xx[n], yy[n], Z[i], cmap=colormap)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cm_bright,edgecolor='k')	
		if first_iter: plt.title(datasets_names[2])

		k+=1; i+=1
		plt.subplot(3,6,k) 
		plt.pcolormesh(xx[n], yy[n], Z[i], cmap=colormap)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cm_bright,edgecolor='k')	
		####### Support vectors in plot
		x = np.linspace(xx[n].min(), xx[n].max(), 30)
		y = np.linspace(yy[n].min(), yy[n].max(), 30)
		X_, Y_ = np.meshgrid(x,y)
		xy = np.vstack([X_.ravel(), Y_.ravel()]).T
		svc = SVC(gamma = Gamma, C = c)
		svc.fit(X[n], y_label[n])
		P = svc.decision_function(xy).reshape(X_.shape)
		plt.contour(X_, Y_, P, colors='k', levels = [-1,0,1], alpha=.5, linestyles=['--', '-', '--'])
		#######
		if first_iter: plt.title(datasets_names[3])

		k+=1; i+=1
		plt.subplot(3,6,k) 
		plt.pcolormesh(xx[n], yy[n], Z[i], cmap=colormap)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cm_bright,edgecolor='k')	
		if first_iter: plt.title(datasets_names[4])

		k+=1; i+=1
		plt.subplot(3,6,k) 
		plt.pcolormesh(xx[n], yy[n], Z[i], cmap=colormap)
		plt.scatter(X[n][:, 0], X[n][:, 1], c=y_label[n], s=7, cmap=cm_bright,edgecolor='k')	
		if first_iter: plt.title(datasets_names[5])

		first_iter = False

	#plt.tight_layout()		
	plt.show()
	