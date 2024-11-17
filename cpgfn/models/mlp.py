from __future__ import annotations
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from typing_extensions import Self,deprecated
from .base import BaseBackboneModule

class OneHotBackbone(BaseBackboneModule):
    '''
    Args:
    
    Note:
    `max_length` should be shared from `Env`.
    
    '''
    def __init__(self,
        embedding_dim:int=1024,
        num_layers:int=5,
        hide_dim:int=2048,
        dropout:float=0.1,
        max_length:int=20,
        ):
        # hard-code s0/sf since they are also hard-coded in Env
        s0_code=20
        sf_code=21
        
        super().__init__(embedding_dim=embedding_dim)
        num_tokens=s0_code
        self.max_length,self.num_tokens,self.sf_code = (
                max_length,num_tokens,sf_code)
        
        (self.hide_dim,self.num_layers,self.dropout
            )=hide_dim,num_layers,dropout
        
        self.input = nn.Linear((self.max_length+2)*(self.num_tokens+1),hide_dim)
        hidden_layers = []
        
        for _ in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(hide_dim, hide_dim))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hide_dim, self.embedding_dim)

    def forward(self, states_tensor:Tensor):
        # print(self.device)
        # print(states_tensor.device)
        x=torch.zeros(*states_tensor.shape,self.num_tokens+1).to(states_tensor.device)
        mask=states_tensor!=self.sf_code
        x[mask]=F.one_hot(states_tensor[mask],self.num_tokens+1).type_as(x)
        x=x.reshape(*x.shape[:-2],-1)
        return self.output(self.hidden(self.input(x)))
    
class PosToFillBackbone(BaseBackboneModule):
    def __init__(self, 
        embedding_dim,
        max_length,
        num_layers:int=5,
        hide_dim:int=2048):
        super().__init__(embedding_dim)
    
@deprecated('use `BaseBackboneModule`s')
class MLPModule(nn.Module):
    def __init__(self,
        num_outputs:int,
        share_trunk_with:Self|None=None,
        s0_code:int=20,
        sf_code:int=-100,
        max_length:int=20,
        hide_dim:int=2048,
        num_layers:int=5,
        dropout:float=0.1,
        ):
        '''
        TODO
        `s0_code` to be curated.
        check `CircularEncoder` for details
        
        `share_trunk_with` would  override every parameter after it.
        '''
        super().__init__()
        
        if share_trunk_with is None:
            num_tokens=s0_code
            self.max_length,self.num_tokens,self.sf_code = (
                    max_length,num_tokens,sf_code)
            
            (self.num_outputs,self.hide_dim,
            self.num_layers,self.dropout
                )=num_outputs,hide_dim,num_layers,dropout
            
            self.input = nn.Linear(self.max_length*(self.num_tokens+1),hide_dim)
            hidden_layers = []
            for _ in range(num_layers):
                hidden_layers.append(nn.Dropout(dropout))
                hidden_layers.append(nn.ReLU())
                hidden_layers.append(nn.Linear(hide_dim, hide_dim))
            self.hidden = nn.Sequential(*hidden_layers)
        else:
            self.max_length,self.num_tokens,self.sf_code=(
                share_trunk_with.max_length,
                share_trunk_with.num_tokens,
                share_trunk_with.sf_code
            )
            (self.num_outputs,self.hide_dim,
            self.num_layers,self.dropout
                )=(share_trunk_with.num_outputs,
                   share_trunk_with.hide_dim,
                   share_trunk_with.num_layers,
                   share_trunk_with.dropout
                   )
            self.input=share_trunk_with.input
            self.hidden=share_trunk_with.hidden
            
        self.output = nn.Linear(hide_dim, num_outputs)
        
    def forward(self, states_tensor:Tensor):
        # states_tensor=torch.where(states_tensor!=self.sf_code,states_tensor,self.num_tokens+1)
        # states_tensor=F.one_hot(states_tensor,num_classes=self.num_tokens+2)
        x=torch.zeros(*states_tensor.shape,self.num_tokens+1).to(states_tensor.device)
        mask=states_tensor!=self.sf_code
        x[mask]=F.one_hot(states_tensor[mask],self.num_tokens+1).type_as(x)
        x=x.reshape(*x.shape[:-2],-1)
        return self.output(self.hidden(self.input(x)))
    
SimplestModule=MLPModule