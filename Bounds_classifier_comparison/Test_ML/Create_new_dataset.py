#Create new dataset

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

#Dataset1
############################################################################

X, y = make_blobs(n_samples=1500, centers=3, random_state=3, cluster_std = .8)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)
#df = pd.DataFrame(X, columns = ['x1', 'x2'])
df = pd.DataFrame({'x1' : X[:,0],
					'x2' : X[:,1],
					'type' : y})
print(df)

#Dataset2
############################################################################
'''
X, y = make_blobs(n_samples=800, centers=8, cluster_std = .5, random_state= 2)
#plt.scatter(X[:,0], X[:,1], c = y, cmap = 'spring')
#plt.show()
#df = pd.DataFrame(X, columns = ['x1', 'x2'])
df = pd.DataFrame({'x1' : X[:,0],
					'x2' : X[:,1],
					'type' : y})
print(df)
'''

#Dataset3
############################################################################
'''
X, y = make_blobs(n_samples=1000, centers=5, cluster_std = .8, random_state= 6)
#plt.scatter(X[:,0], X[:,1], c = y, cmap = 'spring')
#plt.show()
#df = pd.DataFrame(X, columns = ['x1', 'x2'])
df = pd.DataFrame({'x1' : X[:,0],
					'x2' : X[:,1],
					'type' : y})
print(df)
'''

#Dataset4
############################################################################
# mean and standard deviation for the x belonging to the first class
'''
mu_x1, sigma_x1 = 0, 0.1

d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 100) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 100) + 1,
                   'type': 0})

d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 100) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 100) - 1,
                   'type': 1})

d3 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 100) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 100) - 1,
                   'type': 2})

d4 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 100) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 100) + 1,
                   'type': 3})

data = pd.concat([d1, d2, d3, d4], ignore_index=True)
'''
############################################################################
ax = sns.scatterplot(x="x1", y="x2", hue="type", data=df)
plt.show()

#print(data)
df.to_csv('Data/dataset_classifiers1.csv')

