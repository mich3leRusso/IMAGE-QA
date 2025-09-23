import torch 
import torch.nn as nn
from transformers import Swinv2Model
from torchview import draw_graph
import graphviz
from transformers import Swinv2Config, Swinv2Model

graphviz.set_jupyter_format('png')

class Swin_regression(nn.Module):
    """
        This class adapts the Swin Transformer for the regression task 
    """
    def __init__(self,model_name="microsoft/swinv2-tiny-patch4-window8-256",train=True):
        super().__init__()

        # Load pretrained SwinV2
        self.backbone = Swinv2Model.from_pretrained(model_name)
        
        
    
        hidden_size = self.backbone.config.hidden_size
        

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.freeze()

    def forward(self,batch):
        

        embedding=self.backbone(batch).pooler_output
        
        output=self.reg_head(embedding)
        
        return output 
    
    def freeze(self):
        
        for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
         
        for param in self.backbone.parameters():
                param.requires_grad = True
        

         
        
