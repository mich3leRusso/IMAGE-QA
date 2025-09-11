import torch 
from transformers import Swinv2Config, Swinv2Model

# Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
configuration = Swinv2Config()

# Initializing a model (with random weights) from the microsoft/swinv2-tiny-patch4-window8-256 style configuration
model = Swinv2Model(configuration)

