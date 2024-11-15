'''
TODO split backbone and heads
'''
from torch.nn import Module
from abc import ABC
from typing_extensions import Self,deprecated

@deprecated('use `BaseBackboneModule`s')
class BaseSharetrunkModule(Module,ABC):
    def __init__(self, share_trunk_with:Self):
        super().__init__()

class BaseBackboneModule(Module,ABC):
    '''
    Args:
        `embedding_dim`: output dimension of the BackBone Module.  
    '''
    def __init__(self, embedding_dim:int):
        self.embedding_dim = embedding_dim
        super().__init__()