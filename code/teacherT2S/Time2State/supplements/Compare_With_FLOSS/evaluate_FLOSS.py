

   

import os

import json

import pandas as pd

import numpy as np

from TSpy.eval import evaluate_clustering

                            

dataset_list = ['UCR-SEG', 'ActRecTut', 'synthetic_data', 'MoCap', 'USC-HAD', 'PAMAP2']

                      

                                                                               

                                       

                                            

                  

metric='euclidean'

script_path = os.path.dirname(__file__)

                                           

crosscount_save_path = os.path.join(script_path, 'output_FLOSS-%s/crosscount/'%(metric))

clustering_result_save_path = os.path.join(script_path, 'output_FLOSS-%s/clustering'%(metric))

def evaluate(dataset):

    cps_save_path = os.path.join(script_path, 'extracted_seg_pos/%s_result.json'%(dataset))

    if not os.path.exists(cps_save_path):

        return

    with open(cps_save_path, 'r') as f:

        cps_json = json.load(f)

    name_list = list(cps_json)

    ari_list = []

    nmi_list = []

    for file_name in name_list:

        groundtruth = pd.read_csv(os.path.join(script_path, 'data/FLOSS_format/%s/%s.label'%(dataset, file_name[:-4])), header=None).to_numpy()

        prediction = np.load(os.path.join(clustering_result_save_path, '%s/%s.npy'%(dataset, file_name[:-4])))

        ari, anmi, nmi = evaluate_clustering(groundtruth.flatten(), prediction.flatten())        

        ari_list.append(ari)

        nmi_list.append(nmi)

    print('%s: Average ARI is %f, Average NMI is %f'%(dataset, np.mean(ari_list), np.mean(nmi_list)))

for dataset in dataset_list:

    evaluate(dataset)
