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
dataset=None
if dataset_name== "kadid10k":
    #open the dataset
    files=os.listdir(dataset_path)
    print(files)
   # label_file=""
    img_noise=[]
    for file in files:

        if file.endswith(".csv"):
            
            labels_dataset=pd.read_csv(f"{dataset_path}/{file}")   
            img_noise=labels_dataset["dist_img"].values().tolist()
            labels=labels_dataset["dmos"].values().tolist()
        
        else:
            images_name=f"{dataset_path}/{file}"

   #take the important images 
   for image in img_noise 
       
if dataset_name=="LIVE":
    files=os.listdir(dataset_path)
    print(files)

else:
    print("no other dataset at the moment")   