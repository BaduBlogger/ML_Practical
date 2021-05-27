import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = [[0.1,0.6],
	[0.15,0.71],
	[0.08,0.9],
	[0.16, 0.85],
	[0.2,0.3],
	[0.25,0.5],
	[0.24,0.1],
	[0.3,0.2]]

colmap = {1: 'r', 2: 'b'}
R1 = np.array(X)[:,0]
R2 = np.array(X)[:,1]
plt.scatter(R1,R2, color='k')
plt.show()

centroids = np.array([[0.1,0.6],[0.3,0.2]])
print('Intial Centroids :\n',centroids)

C_x=np.array([0.1,0.6])
C_y=np.array([0.3,0.2])
plt.scatter(C_x[0],C_y[0], color=colmap[1])
plt.scatter(C_x[1],C_y[1], color=colmap[2])
plt.show()

plt.scatter(R1,R2, c='#050505')
plt.scatter(C_x[0], C_y[0], marker='*', s=200, c='r')
plt.scatter(C_x[1], C_y[1], marker='*', s=200, c='b')
plt.show()

from sklearn.cluster import KMeans
model = KMeans(n_clusters=2,init=centroids,n_init=1)
model.fit(X)
print('Labels :',model.labels_)

print('Ans 1 -> P6 belongs to cluster :',model.labels_[5])

print('Ans 2 -> No. of population around cluster 2 :',np.count_nonzero(model.labels_==1))

print('Ans 3 -> New Centroids:\n',model.cluster_centers_)