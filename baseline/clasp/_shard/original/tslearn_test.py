from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walks
import numpy as np

X = random_walks(n_ts=50, sz=32, d=1)
print(X.shape)

data1 = np.array([1,2,3,4,5,6])
data2 = np.array([1,2,3,4,5,6])
data3 = np.array([1,2,3,4,5,6])
data4 = np.array([1,2,3,4,5,6])
data5 = np.array([1,2,3,4,5,6])
data6 = np.array([1,2,3,4,5,6])









data1 = np.ones((100,))
data = np.hstack([data1, data1, data1, data1])
print(data.shape)






