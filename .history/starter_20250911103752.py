import torch 
from transformers import Swinv2Config, Swinv2Model
from zipfile import Zip
dataset_name="KADID10"
dataset_path=f"/archive/HPCLab_exchange/MORTE_AL_DAVINCI/{dataset_name}"

#import the model 

model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
#visualize the model

print(model)

if dataset_name== "KADID10":
    #open the dataset
    with unzip

else:
    print("no other dataset at the moment")   