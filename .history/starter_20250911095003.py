import torch 
from transformers import pipeline

pipeline=pipeline(
    task="image-classification"
)