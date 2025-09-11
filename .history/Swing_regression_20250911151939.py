import torch 
import torch.nn as nn
from transformers import Swinv2Model

class Swin_regression(nn.Module):
    """
        This class adapts the Swin Transformer for the regression task 
    """