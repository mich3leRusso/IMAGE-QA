import torch 
from transformers import pipeline
from transf
pipeline=pipeline(
    task="image-classification", 
    model="microsoft/swin-tiny-patch4-window7-224",
    dtype=torch.float16,
    device=0
)