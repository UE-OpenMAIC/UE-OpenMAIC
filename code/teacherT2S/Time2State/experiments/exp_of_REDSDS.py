                                                                         

 

                                                                 

                                                                  

                                     

 

                                                

 

                                                                           

                                                                           

                                                                         

                                                

import os

import json

import torch

import torch.nn as nn

import argparse

import numpy as np

import matplotlib

import sys

sys.path.append('Baselines/REDSDS')

import src.utils as utils

import src.datasets as datasets

import src.tensorboard_utils as tensorboard_utils

from src.model_utils import build_model

from src.evaluation import evaluate_segmentation

from src.torch_utils import torch2numpy

available_datasets = {"bouncing_ball", "3modesystem", "bee"}

                                                        

                   

                      

                                          

                          

                                                                  

                         

                    

                                     

                                                                

                                                    

                                                

                                                             

                                                 

                           

                     

                

                                         

                                            

                                          

                       

       

                        

                                                                                   

       

            

               

                                         

                          

                                       

                                     

                                         

       

                          

                                                                            

                 

                      

                                     

                       

                                                

                                          

                                       

                   

                                      

                                                        

                                                                      

                                                                                       

                                                         

                                                                            

                     

                              

                                                                                   

                                                

                                                          

                           

                                 

                                                    

                                          

                                     

                                           

                             

       

                                                                       

                        

                                                                             

       

                                                            

                                                                             

       

                                                                       

                        

                                                                              

       

                                                              

                           

                                                

                                          

       

                                                                       

                        

                                                                                

       

def get_dataset(dataset):

    assert dataset in available_datasets, f"Unknown dataset {dataset}!"

    if dataset == "bouncing_ball":

        train_dataset = datasets.BouncingBallDataset(path="./data/bouncing_ball.npz")

        test_dataset = datasets.BouncingBallDataset(

            path="./data/bouncing_ball_test.npz"

        )

    elif dataset == "3modesystem":

        train_dataset = datasets.ThreeModeSystemDataset(path="./data/3modesystem.npz")

        test_dataset = datasets.ThreeModeSystemDataset(

            path="./data/3modesystem_test.npz"

        )

    elif dataset == "bee":

        train_dataset = datasets.BeeDataset(path="./data/bee.npz")

        test_dataset = datasets.BeeDataset(path="./data/bee_test.npz")

    return train_dataset, test_dataset

if __name__ == "__main__":

    matplotlib.use("Agg")

                       

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--config", type=str, help="Path to config file.")

    group.add_argument("--ckpt", type=str, help="Path to checkpoint file.")

    parser.add_argument(

        "--device",

        type=str,

        default="cpu",

        help="Which device to use, e.g., cpu, cuda:0, cuda:1, ...",

    )

    args = parser.parse_args()

            

    if args.ckpt:

        ckpt = torch.load(args.ckpt, map_location="cpu")

        config = ckpt["config"]

    else:

        config = utils.get_config_and_setup_dirs(args.config)

    device = torch.device(args.device)

    with open(os.path.join(config["log_dir"], "config.json"), "w") as fp:

        json.dump(config, fp)
