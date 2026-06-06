import numpy as np

import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

matplotlib.rcParams['mathtext.default'] = 'regular'

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "gold", "yellow", "blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]

                                                                              

                                  

                                 

x_label = [50, 40, 30, 20, 10, 1]                   

y_label = [1,2,4,6,8,10]

                

x = list(range(len(x_label)))

y = list(range(len(y_label)))

x_tickets = [str(_x) for _x in x_label]

y_tickets = [str(_x) for _x in y_label]

acc = [[0.582,0.5886,0.5904,0.58,0.5735, 0.5756],

[0.6388, 0.6427, 0.6531, 0.6418, 0.6364, 0.4525],

[0.6528, 0.6586, 0.6472, 0.6526, 0.6411, 0.4551],

[0.6657, 0.6636, 0.6624, 0.6648, 0.6553, 0.4588],

[0.6612, 0.6535, 0.6674, 0.6480, 0.6525, 0.4691],

[0.6463, 0.6563, 0.6525, 0.6583, 0.6374, 0.4564]]

acc = np.array(acc).T

              

                                                    

xx, yy = np.meshgrid(x, y)                               

                                                  

color_list = []

for i in range(len(y)):

    c = COLOR[i]

    color_list.append([c] * len(x))

color_list = np.asarray(color_list)

                   

                                                      

xx_flat, yy_flat, acc_flat, color_flat = (
    xx.ravel(), yy.ravel(), acc.T.ravel(), color_list.ravel()
)

                

                

                                         

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.bar3d(xx_flat - 0.35, yy_flat - 0.35, 0, 0.7, 0.7, acc_flat,

    color=color_flat,      

    edgecolor="black",        

    shade=True)       

      

ax.set_xlabel(r"$M$")

ax.set_ylabel(r"$N$")

ax.set_zlabel("ARI")

       

ax.set_zlim((0, 1.01))

         

                                        

                                                                           

ax.set_xticks(x)

ax.set_xticklabels(x_tickets)

ax.set_yticks(y)

ax.set_yticklabels(y_tickets)

    

                    

                                                             

                

plt.show()
