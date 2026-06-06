import matplotlib.pyplot as plt

import numpy as np

            

                                              

                                             

                                               

                                               

                                                      

                                                 

                 

                                                           

                                                             

                                                                  

                                                         

                                                           

                                                          

           

                                                             

                                                                  

                                                         

                                                           

                                                           

def ARI():

    plt.figure(figsize=(10, 3.5))

    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']

    Time2State =  [0.8176, 0.7812, 0.8119, 0.3236, 0.6048, 0.4321]

    FLOSS_EUCLIDEAN      =  [0.1230, 0.3937, 0.1921, 0.0343, 0.1984, 0.9169]

    FLOSS_DTW      =  [0.1052, 0.4046, 0.2379, 0, 0.1967, 0.9444]

    x = np.arange(len(labels))                   

    width = 0.2                

    plt.style.use('ggplot')

    plt.bar(x - width, Time2State, width, label='Time2State', hatch='\\.')

    plt.bar(x , FLOSS_EUCLIDEAN, width, label='FLOSS+TSKMeans-euclidean', hatch='\\\.')

    plt.bar(x + width, FLOSS_DTW, width, label='FLOSS+TSKMeans-dtw', hatch='///.')

    plt.ylabel('ARI', size=15)

    plt.xticks(x, labels=labels, size=15)

    plt.yticks(size=15)

    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=15, loc='upper center')

    plt.tight_layout()

    plt.show()

def NMI():

    plt.figure(figsize=(10, 3.5))

    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']

    Time2State =  [0.8268, 0.7943, 0.7634, 0.6355, 0.7956, 0.4848]

    FLOSS_EUCLIDEAN      =  [0.2446, 0.5602, 0.2839, 0.2290,  0.5428, 0.9056]

    FLOSS_DTW =  [0.2758, 0.5611, 0.3461, 0, 0.5285, 0.9246]

    x = np.arange(len(labels))                   

    width = 0.25                

    plt.style.use('ggplot')

    plt.bar(x - width, Time2State, width, label='Time2State', hatch='\\.')

    plt.bar(x , FLOSS_EUCLIDEAN, width, label='FLOSS+TSKMeans-euclidean', hatch='\\\.')

    plt.bar(x + width, FLOSS_DTW, width, label='FLOSS+TSKMeans-dtw', hatch='///.')

    plt.ylabel('NMI', size=15)

    plt.xticks(x, labels=labels, size=15)

    plt.yticks(size=15)

    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=15, loc='upper center')

    plt.tight_layout()

    plt.show()

ARI()

NMI()
