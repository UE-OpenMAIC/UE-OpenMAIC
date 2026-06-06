

   

import os

import json

from miniutils import create_path

dataset_list = ['UCR-SEG', 'MoCap', 'ActRecTut', 'synthetic_data', 'PAMAP2', 'USC-HAD']

script_path = os.path.dirname(__file__)

metric = 'euclidean'

create_path(os.path.join(script_path, 'extracted_seg_pos'))

def extract_seg_result(dataset):

    scores_path = os.path.join(script_path, 'output_FLOSS-%s/%s_segpos.txt'%(metric, dataset))

    if not os.path.exists(scores_path):

        return

    with open(scores_path) as f:

                                            

                                                      

        lines = f.readlines()

    result_json = {}

    for line in lines:

                                          

        data_name = line.split(',')[0].strip()

                                                  

        seg_result_str = line.split(',')[1].strip()

        seg_result_list = [int(seg_pos) for seg_pos in seg_result_str.split('_')]

        result_json[data_name] = seg_result_list

    print(result_json)

    with open(os.path.join(script_path, 'extracted_seg_pos/%s_result.json'%(dataset)), 'w') as f:

        f.write(json.dumps(result_json))

        f.close()

for dataset in dataset_list:

    extract_seg_result(dataset)
