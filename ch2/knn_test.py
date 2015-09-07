import matplotlib
import matplotlib.pyplot as plt
from numpy import array
import knn

datingDataMat,datingLabels,vector = knn.file2matrix('datingTestSet.txt')

fig = plt.figure()
ax = fig.add_subplot(131)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax = fig.add_subplot(132)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(vector),15.0*array(vector))
ax = fig.add_subplot(133)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(vector),15.0*array(vector))
plt.show()
