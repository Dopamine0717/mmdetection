import matplotlib.pyplot as plt
import matplotlib
<<<<<<< HEAD
matplotlib.use('TkAgg')
=======
matplotlib.use('TkAgg')
>>>>>>> 3e64bc4b0e3d7e97beaabecda03716d5f17557ef
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')
ax.scatter(np.random.rand(10),np.random.rand(10),np.random.rand(10))
plt.show()