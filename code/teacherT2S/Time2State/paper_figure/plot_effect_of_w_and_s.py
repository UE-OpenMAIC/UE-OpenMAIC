import numpy as np

import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

matplotlib.rcParams['mathtext.default'] = 'regular'

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow", "blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]

                                                                              

x_label = np.arange(10, 100+1, 10)

y_label = np.arange(100, 1000+1, 100)

                

x = list(range(len(x_label)))

y = list(range(len(y_label)))

x_tickets = [str(_x) for _x in x_label]

y_tickets = [str(_x) for _x in y_label]

                                      

                                                              

                       

acc = [[0.6507, 0.6398, 0.6371, 0.6016, 0.5998, 0.5753, 0.5647, 0.5553, 0.5627, 0.5696],

[0.6502, 0.6601, 0.6419, 0.6377, 0.6276, 0.6110, 0.6019, 0.5844, 0.5727, 0.5660],

[0.6582, 0.6624, 0.6479, 0.6402, 0.6339, 0.6349, 0.6170, 0.6150, 0.6073, 0.6013],

[0.6838, 0.6726, 0.6770, 0.6639, 0.6634, 0.6395, 0.6229, 0.6252, 0.6103, 0.6219],

[0.6498, 0.6534, 0.6498, 0.6357, 0.6248, 0.6185, 0.6105, 0.5980, 0.6026, 0.6039],

[0.6336, 0.6367, 0.6267, 0.6295, 0.6231, 0.6102, 0.6003, 0.6036, 0.5922, 0.5907],

[0.6262, 0.6270, 0.6285, 0.6190, 0.6104, 0.6060, 0.6001, 0.5960, 0.5796, 0.5830],

[0.6323, 0.6311, 0.6247, 0.6273, 0.6266, 0.6151, 0.6168, 0.6068, 0.6040, 0.6052],

[0.6323, 0.6311, 0.6247, 0.6273, 0.6266, 0.6151, 0.6168, 0.6068, 0.6040, 0.6052],

[0.6323, 0.6311, 0.6247, 0.6273, 0.6266, 0.6151, 0.6168, 0.6068, 0.6040, 0.6052]]

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

      

ax.set_xlabel(r"$s$")

ax.set_ylabel(r"$w$")

ax.set_zlabel("ARI")

       

ax.set_zlim((0, 1.01))

         

                                        

                                                                           

ax.set_xticks(x)

ax.set_xticklabels(x_tickets)

ax.set_yticks(y)

ax.set_yticklabels(y_tickets)

    

                    

                                                             

                

plt.show()
