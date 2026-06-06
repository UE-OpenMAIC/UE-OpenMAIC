                                  

                                           

                                                   

                                                          

 

                                                 

   

                                                                         

                                                           

                              

import numpy as np

import matplotlib.pyplot as plt

import re

import os

import numpy as np

import pandas as pd

def retrieve(in_path, out_path, usecols, show=False):

    cols = {}

    cols_numeric = {}

    for col_name in usecols:

        cols[col_name]=[]

    with open(in_path) as f:

        lines = f.readlines()

        for line in lines:

            for col_name in usecols:

                if re.match(col_name, line):

                    cols[col_name].append(line)

    for col_name in usecols:

        cols_numeric[col_name]=np.array([float(line.split(' ')[1]) for line in cols[col_name]])

    

    if show:

        for col_name in usecols:

            plt.plot(cols_numeric[col_name])

            plt.title(in_path+str(len(cols_numeric[usecols[1]])))

        plt.show()

    

    df_array = []

    for col_name in usecols:

        df_array.append(cols_numeric[col_name])

    df = pd.DataFrame(df_array).T.round(4)

    df.to_csv(out_path, header=None, index=None, sep=' ')

usecols = ['lhumerus','rhumerus','lfemur','rfemur']

out_dir = './4d/'

in_dir = './raw/'

if not os.path.exists(out_dir):

    os.mkdir(out_dir)

                              

                         

                                      

                     

                                                       

retrieve('./raw/amc_86_10.txt', './4d/amc_02_01.4d', usecols, show=True)
