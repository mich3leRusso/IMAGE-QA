import torch 
from transformers import Swinv2Config, Swinv2Model
dataset_name="KADID10"
dataset=F"/archive/HPCLab_exchange/MORTE_AL_DAVINCI/KADID10"

model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
print(model)