import torch 
from transformers import pipeline
from transformers import SwinConfig, SwinModel

pipeline=pipeline(
    task="image-classification", 
    model="microsoft/swin-tiny-patch4-window7-224",
    dtype=torch.float16,
    device=0
)

config=SwinConfig()
model=SwinModel(config)

from transformers import Swinv2Config, Swinv2Model

# Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
configuration = Swinv2Config()

# Initializing a model (with random weights) from the microsoft/swinv2-tiny-patch4-window8-256 style configuration
model = Swinv2Model(configuration)

# Accessing the model configuration
configuration = model.config