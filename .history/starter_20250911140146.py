import torch 
from transformers import Swinv2Config, Swinv2Model
import os
import pandas as pd 
dataset_name="kadid10k"
dataset_path=f"/archive/HPCLab_exchange/MORTE_AL_DAVINCI/KADID10/{dataset_name}"

#import the model 

model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
#visualize the model

print(model)


labels=None
if dataset_name== "kadid10k":
    #open the dataset
    files=os.listdir(dataset_path)
    print(files)
   # label_file=""
    for file in files:
        if file.endswith(".csv"):
            labelspd.read_csv(f"{dataset_path}/{file}","r")        
    


else:
    print("no other dataset at the moment")   