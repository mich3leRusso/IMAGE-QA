import torch 
from transformers import pipeline
from transformers import SwinConfig, SwinModel

pipeline=pipeline(
    task="image-classification", 
    model="microsoft/swin-tiny-patch4-window7-224",
    dtype=torch.float16,
    device=0
)

confiurat