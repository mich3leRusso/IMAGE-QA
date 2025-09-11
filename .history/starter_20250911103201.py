import torch 
from transformers import Swinv2Config, Swinv2Model

dataset="/archive/HPCLab"
model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
print(model)