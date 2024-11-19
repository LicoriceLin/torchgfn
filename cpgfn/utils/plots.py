from typing import List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import Tensor

def seqlength_dist(seqs:List[str],max_length:int=20):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    pd.Series([len(i)for i in seqs]).hist(ax=ax,density=True,bins=np.linspace(0,max_length,max_length+1))
    ax.set_xticks(np.linspace(0,max_length,max_length+1),np.linspace(0,max_length,max_length+1))
    return fig,ax

aa_tokens = tuple('KREDHQNCGPASTMLVIFWY')
def aa_dist(seqs:List[str],max_length:int=20):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    array=np.zeros((max_length,20),dtype=np.int64)
    for seq in seqs:
        for i,aa in enumerate(seq):
            array[i,aa_tokens.index(aa)]+=1
    
    array=(array/((array.sum(axis=1)+1e-10
            ).reshape(-1,1))).T
    np.nan_to_num(array,nan=0.)
    '''
    TODO
    vmax=1 might be too large?
    '''
    ax.imshow(array,cmap='YlGn',vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0,max_length),np.arange(1,max_length+1))
    ax.set_yticks(np.arange(0,20),aa_tokens)
    ax.set_xlabel('aa position')
    ax.set_ylabel('aa type')
    return fig,ax

def reward_dist(log_rewards:Tensor,max_score:int=20):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    ax.hist(log_rewards.reshape(-1).to('cpu').numpy(),
        bins=np.linspace(0,max_score,max_score+1),density=True)
    return fig,ax


mode_dict={
        0:'KREDHQN',
        1:'CGPAST',
        2:'MLVIFWY',
    }
aa_group_map={ a:k for k,v in mode_dict.items() for a in v}

def simple_aagroup_dist(seqs:List[str],max_length:int=20):
    fig,ax=plt.subplots(1,1)
    ax:Axes
    array=np.zeros((max_length,3),dtype=np.int64)
    for seq in seqs:
        for i,aa in enumerate(seq):
            array[i,aa_group_map[aa]]+=1
    
    array=(array/((array.sum(axis=1)+1e-10
            ).reshape(-1,1))).T
    np.nan_to_num(array,nan=0.)
    ax.imshow(array,cmap='YlGn',vmin=0., vmax=1.)
    ax.set_xticks(np.arange(0,max_length),np.arange(1,max_length+1))
    ax.set_yticks(np.arange(0,3),[0,1,2])
    ax.set_xlabel('aa position')
    ax.set_ylabel('aa group')
    return fig,ax