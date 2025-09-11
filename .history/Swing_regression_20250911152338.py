import torch 
import torch.nn as nn
from transformers import Swinv2Model

class Swin_regression(nn.Module):
    """
        This class adapts the Swin Transformer for the regression task 
    """
    def __init__(self,model_name="microsoft/swinv2-tiny-patch4-window8-256",train=True):
        self.__init__()

        # Load pretrained SwinV2
        self.backbone = Swinv2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Optional: freeze backbone weights for small datasets
        if not train:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head: maps Swin features to one value
        self.reg_head = nn.Linear(hidden_size, 1)

    def foward(self,batch):
        embeddingself.backbone(batch)
        sel
        
        return results 