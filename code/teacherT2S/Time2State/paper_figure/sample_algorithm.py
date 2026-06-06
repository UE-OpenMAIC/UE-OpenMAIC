import matplotlib.pyplot as plt

from TSpy.utils import *

def plot_example():

    import os

    import scipy.io

    import pandas as pd

    from TSpy.utils import normalize

    from TSpy.eval import find_cut_points_from_label

    

                                                  

                       

                                     

                      

                                                                                                     

                                                                              

            

                                  

    

    data_path = os.path.join(os.path.dirname(__file__), '../data/ActRecTut/subject2_gesture/data.mat')

    data = scipy.io.loadmat(data_path)

    X = data['data'][22270:23800,:6]

                      

    groundtruth = data['labels'].flatten()[22270:23800]

                                                                                                   

                                                       

                          

                       

                                                 

                               

    plt.style.use('bmh')

    fig, ax = plt.subplots(figsize=(8,1.2))

    plt.xticks(color='w')

    plt.yticks(color='w')

    ax.axes.tick_params(size=0)

    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    normalize(X)

                                                               

                                                            

    plt.plot(X[:,4], linewidth=1, alpha=0.9)

                

                                 

                                                  

                                                            

                              

                                                               

                            

                                 

                         

                        

                                          

                                                                     

                              

                                      

                                 

                    

                                                                          

                                    

                           

                               

                                                             

    plt.tight_layout()

    plt.show()

plot_example()
