from __future__ import annotations
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from typing_extensions import Self,deprecated
from .base import BaseBackboneModule
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Literal

class CircularEncoder(BaseBackboneModule):
    '''

    TODO
    1.Our module is now 100% autoregressive.
        Use Decoder-Only frameworks.
    
    '''
    def __init__(self, 
            embedding_dim:int=128,
            nhead:int = 4,
            dim_feedforward = 512,
            max_length:int=20,
            num_layers:int=1,
            pos_embedding:Literal['circular','linear']='circular'
            ):
        # hard-code s0/sf since they are also hard-coded in Env
        s0_code:int=20
        sf_code:int=21
        n_tokens:int=22 
        # for robustness, take <pad> into consideration.
        # 20 aa + <bos> + <pad>
        # TODO 
        # due to silly setting of pgfn, I cannot add a <eos> alongside with exit actions.
        super().__init__(embedding_dim=embedding_dim)
        
        (self.sf_code, self.s0_code, self.max_length) = (
            sf_code,s0_code,max_length,)
        (self.dim_feedforward,self.nhead,self.num_layers)=(
            dim_feedforward,nhead,num_layers)
        assert self.embedding_dim % (self.nhead * 2) == 0, "invalid nhead"
        self.pos_eb_dim = self.embedding_dim // (self.nhead * 2)
        
        self.embedding = nn.Embedding(
            num_embeddings=n_tokens,
            embedding_dim=self.embedding_dim,
            padding_idx=s0_code,
        )
        
        encoder = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )
        self.encoder=nn.TransformerEncoder(encoder,num_layers=self.num_layers)
        
        self.pos_embedding=pos_embedding
        if self.pos_embedding=='circular':
            self.pos_embedder=self.circular_embedding
        elif self.pos_embedding=='linear':
            self.pos_embedder_=RotaryPositionalEmbeddings(
                dim=self.embedding_dim//self.nhead,
                max_seq_len=self.max_length+2)
            self.pos_embedder=self.linear_embedding
    def forward(
        self, trajs: Tensor
    ) -> Tensor:
        '''
        TODO
        `torchtyping.TensorType` annotations are removed temperally.
        As they are incompatible with static type checkers.
        Consider future implementation of `jaxtyping`

        '''
        if trajs.shape[0]==0:
            return torch.zeros(*trajs.shape,self.embedding_dim,device=trajs.device)
        
        # trajs = torch.where(trajs != self.sf_code, trajs, self.s0_code)
        # trajs.clone()
        # trajs[trajs == self.sf_code] = self.s0_code
        batch_shape, state_shape = trajs.shape[:-1], trajs.shape[-1]
        ori_mask = trajs == self.sf_code
        
        ori_mask[...,0] = False 
        
        trajs = trajs.view(-1, state_shape)
        # src_key_padding_mask= ori_mask.view(-1,state_shape)

        trajs = self.embedding(trajs)
        
        encoder_mask = ori_mask.view(-1, state_shape)
        # print(self.positional_embedding(encoder_mask).shape)
        trajs=self.pos_embedder(trajs,encoder_mask)
        
        trajs = self.encoder(src=trajs, src_key_padding_mask=encoder_mask)
        trajs = trajs.view(batch_shape + trajs.shape[-2:])

        # src_key_padding_mask_dim=len(ori_mask.shape)
        # ext_key_padding_mask = ori_mask.reshape(*ori_mask.shape, 1).repeat(
        #     *[1] * len(ori_mask.shape), trajs.shape[-1]
        # )
        # print(ext_key_padding_mask.shape)
        # print(trajs.shape)
        # trajs[ext_key_padding_mask] = torch.nan
        # return torch.nanmean(trajs, dim=-2)
        # print(trajs[...,0,:].shape)
        # print(torch.nanmean(trajs, dim=-2).shape)
        return trajs[...,0,:]

    def linear_embedding(self, trajs:Tensor, encoder_mask:Tensor):
        b,s,n=trajs.shape
        trajs=trajs.reshape(b,s,self.nhead,self.embedding_dim//self.nhead)
        trajs=self.pos_embedder_(trajs)
        return trajs.reshape(b,s,-1)
        # assert isinstance(self.embedder,RotaryPositionalEmbeddings)
        
    def circular_embedding(self, trajs:Tensor,encoder_mask:Tensor):
        b, l, e = encoder_mask.shape[0], self.max_length+2, self.pos_eb_dim
        valid_length = (~encoder_mask).long().sum(dim=-1)
        _ = torch.einsum(
            "i,j,k->ijk",
            1 / (valid_length + 1e-8),
            torch.arange(0, l).to(valid_length.device),
            2 * torch.pi * torch.arange(0, e).to(valid_length.device),
        )
        ebs = torch.zeros([b, l, e * 2]).to(valid_length.device)
        ebs[:, :, 0::2] = torch.sin(_)
        ebs[:, :, 1::2] = torch.cos(_)

        trajs = trajs + ebs.repeat(1, 1, self.nhead)
        trajs = (trajs) / torch.linalg.vector_norm(trajs, dim=-1, keepdim=True)
        return trajs

@deprecated('use `BaseBackboneModule`s')
class CircularEncoderModule(nn.Module):
    '''
    Here encoder module is forced to initialized before, 
    or shared from `share_trunk_with`, 
    which would override every parameter after it. 
    '''
    def __init__(self, 
                num_outputs: int, 
                share_trunk_with:Self|None=None,
                encoder: CircularEncoder|None=None,
                hide_dim:int=2048,
                dropout:float=0.1):
        super().__init__()
        self.num_outputs = num_outputs
        
        if share_trunk_with is not None:
            self.encoder = share_trunk_with.encoder
            self.hide_dim = share_trunk_with.hide_dim
            self.dropout = share_trunk_with.dropout
        else:
            assert encoder is not None
            self.encoder = encoder
            self.hide_dim = hide_dim
            self.dropout=dropout
        self.head = nn.Sequential(
            *[
                nn.Linear(self.encoder.embedding_dim, self.hide_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(self.hide_dim, self.num_outputs),
                nn.LayerNorm(self.num_outputs),
            ]
        )

    def forward(
        self, trajs: Tensor
    ) -> Tensor:
        trajs = self.encoder(trajs)
        trajs = self.head(trajs)
        # TMP nan to euqal
        # trajs[trajs.isnan()]=1/self.outdim
        trajs = torch.where(~trajs.isnan(), trajs, 1 / self.num_outputs)

        return trajs
    
SillyModule = CircularEncoderModule