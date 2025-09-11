import torch 
import torch.nn as nn
from transformers import Swinv2Model

class Swin_regression(nn.Module):
    """
        This class adapts the Swin Transformer for the regression task 
    """
    def __init__(self,model_name="microsoft/swinv2-tiny-patch4-window8-256",Train=True ):