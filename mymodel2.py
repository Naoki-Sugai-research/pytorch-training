import torch
 from torch import nn

class MyModel(nn.Module):
    def __init__
    (self,
     mytensor,
     elem_add,
     elem_multipy):
    super().__init__()
    self.mytensor = mytensor
    self.elem_add = elem_add
    self.elem_multipy = elem_multipy    

    def forward(self, x):
       