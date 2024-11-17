import torch
from torch import nn,Tensor
from torch.utils.data import DataLoader,Dataset
from torch._prims_common import DeviceLikeType
from Bio.Data.IUPACData import protein_letters_1to3

from gfn.actions import Actions
from gfn.containers import Trajectories
from gfn.env import DiscreteEnv, Preprocessor
from gfn.states import DiscreteStates, States

from typing import List, Literal, Optional, Tuple,Callable,Dict,Any
from functools import partial
from torch import dtype
# import models

import numpy as np
import pandas as pd

from cpgfn import rewards
from inspect import signature,_empty

import random

class IdentityLongPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    def preprocess(self, states: States) -> Tensor:
        return states.tensor.long()

    def __call__(self, states: States) -> Tensor:
        return self.preprocess(states)

    def __repr__(self):
        return f"{self.__class__.__name__}, output_dim={self.output_dim}"
    

# Literal[*tuple(rewards.REWARD_REGISTRY.keys())]
class AdditivePepEnv(DiscreteEnv):
    '''
    
    rigid properties
    
    n_actions: 21 (aa + 1)
    Action IDs:
    0-19: Add AAs
    20: Exits
    21: Dummy Actions
    
    Token IDs
    0-19: AAs 
    20: <bos>; s0_code
    21: <pad>; sf_code
    
    s0: [20,21,21,...]
    sf: [21,21,21,...]
    
    `reward`: key in rewards.REWARD_REGISTRY
    
    '''
    def __init__(
        self,
        reward:str,
        min_length: int = 5,
        max_length: int = 16,
        device_str: str|int = 'cpu',
        reward_kwargs:Dict[str,Any]={},
        preprocessor_mode:Literal['identical','pos_to_fill']='identical'
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.device = torch.device(device_str) # redundant
        self.aa_tokens = tuple(protein_letters_1to3.keys())


        # s_max_length: [20,0,1,...,21] (last code should always be 21) 
        n_actions = len(self.aa_tokens) + 1  # last action reserved for exit

        self.s0_code, self.sf_code, self.dummy_code, self.exit_code = (
            n_actions - 1,
            n_actions,
            n_actions,
            n_actions -1,
        )
        state_shape = (self.max_length+2,)
        s0 = torch.full(
            state_shape, fill_value=self.sf_code, dtype=torch.long, device=self.device
        )
        s0[0]=self.s0_code
        
        sf = torch.full(
            state_shape, fill_value=self.sf_code, dtype=torch.long, device=self.device
        )
        
        

        action_shape = (1,)
        # preprocessor = OneHotPreprocessor(token_size=len(self.aa_tokens))
        if preprocessor_mode=='identical':
            preprocessor = IdentityLongPreprocessor(output_dim=max_length)
        elif preprocessor_mode=='pos_to_fill':
            preprocessor = lambda states:self.cal_first_unfilled_pos(states_tensor=states.tensor)
            
        dummy_action = torch.tensor([self.dummy_code], device=self.device)
        exit_action = torch.tensor([self.exit_code], device=self.device)
        super().__init__(
            n_actions=n_actions,
            action_shape=action_shape,
            state_shape=state_shape,
            s0=s0,
            sf=sf,
            dummy_action=dummy_action,
            exit_action=exit_action,
            preprocessor=preprocessor,
            device_str=device_str,
        )
        
        assert reward in rewards.REWARD_REGISTRY
        reward_fn=rewards.REWARD_REGISTRY[reward]
        si=signature(reward_fn)
        assert 'seq' in si.parameters
        reward_kwargs_={k:v.default for k,v in si.parameters.items() if v.default is not _empty}
        reward_kwargs_.update(reward_kwargs)
        if 'max_length' in si.parameters:
            reward_kwargs_.update({'max_length':max_length})
        self.reward_kwargs=reward_kwargs_
        self.reward_fn=partial(reward_fn,**reward_kwargs)
        
        # self.states_class=self.make_states_class()
        # self.actions_class=self.make_actions_class()
        # self.module_pf,self.module_pb=self.make_modules()
        # tmp offline sample entries
        # self.offline_db = pd.read_csv("5LSO.db.csv").set_index("seq")
        # self.offline_db = self.offline_db[self.offline_db["score"] > 0]
        # self.offline_db["score"] = self.offline_db["score"].apply(
        #     lambda x: 0.02 if x < 0.02 else x / 10
        # )

    def step(
        self, states: DiscreteStates, actions: Actions
    ) -> Tensor:
        """
        only consider valid & non-exit actions here
        """
        states_tensor, action_tensor = states.tensor, actions.tensor.squeeze(-1)

        # last_unfill_pos
        first_unfilled_pos = self.cal_first_unfilled_pos(states_tensor)
        shape=states_tensor.shape
        states_tensor=states_tensor.reshape(-1,shape[-1])
        first_unfilled_pos=first_unfilled_pos.reshape(-1)
        # import pdb;pdb.set_trace()
        states_tensor[torch.arange(states_tensor.shape[0]),
                      first_unfilled_pos]=action_tensor
        states_tensor.reshape(*shape)
        return states_tensor
        # make sure for full-filled states, the only action allowed is exit_action

        # might be redundant? as is secured in `update_masks`
        # assert torch.all(action_tensor[last_unfilled_pos==-1] == self.exit_code)

        # fill the last-unfilled pos with given aa codes/exit code
        # valid_mask= last_unfilled_pos!=-1
        # valid_mask = (
        #     (first_unfilled_pos != -1)
        #     & (action_tensor != self.exit_code)
        #     & (action_tensor != self.dummy_code)
        # )
        # valid_indices = first_unfilled_pos[valid_mask]
        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)
        # return states_tensor,valid_mask,valid_indices
        # output = states_tensor.detach()
        # output[(*valid_mask.nonzero(as_tuple=True), valid_indices)] = action_tensor[
        #     valid_mask
        # ]
        # return output

    def backward_step(
        self, states: DiscreteStates, actions: Actions
    ) -> Tensor:

        states_tensor, action_tensor = states.tensor, actions.tensor.squeeze(-1)
        first_unfilled_pos = self.cal_first_unfilled_pos(states_tensor)
        shape=states_tensor.shape
        
        states_tensor=states_tensor.reshape(-1,shape[-1])
        first_unfilled_pos=first_unfilled_pos.reshape(-1)
        states_tensor[
            torch.arange(states_tensor.shape[0]),
            first_unfilled_pos-1]=self.sf_code
        states_tensor.reshape(*shape)
        return states_tensor
        # current state is not s0
        # valid_mask = last_filled_pos != -1
        # valid_indices = last_filled_pos[valid_mask]

        # batch_indices = torch.arange(states_tensor.shape[0]
        #         )[valid_mask].to(states_tensor.device)'
        # return states_tensor,last_filled_pos,valid_mask,valid_indices
        # assert (states_tensor[(*valid_mask.nonzero(as_tuple=True),valid_indices)] == action_tensor[valid_mask]).all()

        # output = states_tensor.detach()
        # output[(*valid_mask.nonzero(as_tuple=True), valid_indices)] = self.s0_code

        # return output
        # mask = torch.argmax(mask_, dim=1)
        # mask[~mask_.any(dim=1)] = -1
             
    def update_masks(self, states: DiscreteStates) -> None:
        """Update the masks based on the current states."""
        states_tensor = states.tensor
        first_unfilled_pos=self.cal_first_unfilled_pos(states_tensor)
        # last_filled_pos = self.cal_last_filled_pos(states_tensor)

        states.forward_masks = torch.ones(
            (*states.batch_shape, self.n_actions),
            dtype=torch.bool,
            device=states_tensor.device,
        )

        # for full-filled states, only allow exit
        states.forward_masks[
            (
                *(first_unfilled_pos == self.max_length + 1).nonzero(as_tuple=True),
                slice(None, self.n_actions - 1),
            )
        ] = False
        # for sf states, everything prohibited
        states.forward_masks[
            (
                *(first_unfilled_pos == 0).nonzero(as_tuple=True),
                slice(None, self.n_actions),
            )
        ] = False
        # states.forward_masks[(*(last_filled_pos==self.max_length-1).nonzero(as_tuple=True),slice(None,self.n_actions-1))]
        # for l<l_min, prohibit exit actions
        states.forward_masks[
            (
                *(first_unfilled_pos-1 < self.min_length ).nonzero(as_tuple=True),
                self.n_actions - 1,
            )
        ] = False
        # return
        states.backward_masks = torch.zeros(
            (*states.batch_shape, self.n_actions - 1),
            dtype=torch.bool,
            device=states_tensor.device,
        )
        valid_mask = first_unfilled_pos > 1 # states with valid last-
        
        valid_indices=states_tensor[
             (*valid_mask.nonzero(as_tuple=True),
              first_unfilled_pos[valid_mask]-1
              )
            ]
        
        states.backward_masks[(
                *valid_mask.nonzero(as_tuple=True), 
                valid_indices,
            )] = True
        
        # valid_mask = last_filled_pos != -1
        # valid_indices = states_tensor[
        #     (*valid_mask.nonzero(as_tuple=True), last_filled_pos[valid_mask])
        # ]
        # states.backward_masks[
        #     (*valid_mask.nonzero(as_tuple=True), valid_indices)
        # ] = True

    def log_reward(
        self, final_states: DiscreteStates
    ) -> Tensor:
        # place holder
        seqs = self.states_to_seqs(final_states)
        return torch.tensor([self.reward_fn(i) for i in seqs]).to(final_states.tensor.device)
        # if self.reward_mode=='look_up':
        #     return torch.tensor([self.offline_db["score"].get(i, 0.02) for i in seqs]).to(self.device)
        # elif self.reward_mode=='simple_pattern':
        #     '''
        #     debug rewards:
        #     1. seq-length: rewards:
        #     length:
        #         [0,4]: 2*x
        #         [5,9]: 10-2*(x-5)
        #         [10,14]: 2*(x-10)
        #         [15,20]: 10-2*(x-15)
            
        #     1-5 up; 5-15 down; 15-20 up
        #     2. pattern rewards:
        #     switch(pos%3):
        #         0 -> KREDHQN +p;
        #         1 -> CGPAST +p;
        #         2 -> VMLVIFWY +p;
        #         p: pos-wise score = (10/seq-length)
        #     '''
        #     mode_dict={
        #         0:'KREDHQN',
        #         1:'CGPAST',
        #         2:'MLVIFWY',
        #     }
        #     def simple_pattern(seq:str):
        #         s=0.
        #         seq_len=len(seq)
        #         if seq_len in [0,20]:
        #             return s+1e-5
        #         elif seq_len<=4:
        #             s+=2*seq_len
        #         elif seq_len<=9:
        #             s+=10-2*(seq_len-5)
        #         elif seq_len<=14:
        #             s+=2*(seq_len-10)
        #         elif seq_len<20:
        #             s+=10-2*(seq_len-15)
                    
        #         pos_score=10/len(seq)
        #         for i,a in enumerate(seq):
        #             if a in mode_dict[i%3]:
        #                 s+=pos_score
        #         return s+1e-5
        #     return torch.log(torch.tensor([simple_pattern(i) for i in seqs])).to(self.device)
        # else:
        #     raise ValueError
        # return torch.log(torch.tensor([self.offline_db['score'].get(i,0.02) for i in seqs])).to(self.device)
        # return torch.randn(final_states.batch_shape,device=final_states.tensor.device)

    
    ### utilities ###
    def cal_first_unfilled_pos(
        self, states_tensor: Tensor) -> Tensor:
        """
        Return:  
        
        s0: return 1  
        sf: return 0  
        s_full: return max_length+1  
        others: the idx of first sf_codes, i.e. real seq_length + 1  
        """
        first_unfilled_pos_ = (states_tensor == self.sf_code).long()
        first_unfilled_pos = torch.argmax(first_unfilled_pos_, dim=-1)
        return first_unfilled_pos
        # first_unfilled_pos[~first_unfilled_pos_.any(dim=-1)] = (
        #     -1
        # )  # -1 for full-filled states
        # first_unfilled_pos[(states_tensor == self.sf_code).any(dim=-1)] = -1
        # return first_unfilled_pos

    # def cal_last_filled_pos(
    #     self, states_tensor: Tensor) -> Tensor:
    #     """
    #     -1 for s0 states & dummy
    #     """
    #     # get those totally unfilled pos
    #     last_filled_pos = self.cal_first_unfilled_pos(states_tensor) - 1
    #     # no unfilled -> last filled pos = len(states)
    #     last_filled_pos[last_filled_pos == -2] = self.max_length - 1
    #     last_filled_pos[(states_tensor == self.sf_code).any(dim=-1)] = -1
    #     return last_filled_pos
    
    def to(self,device:DeviceLikeType|None=None,dtype:dtype|None=None):
        if device is not None:
            if not isinstance(device,torch.device):
                device=torch.device(device)
            self.device=device
            for k,v in self.__dict__.items():
                # iterative `to`? maybe later.
                if isinstance(v,Tensor):
                    setattr(self,k,v.to(device))
        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()
        #dtype not in use
        return self
    ### reformat ###
    def states_to_seqs(self, final_states: DiscreteStates) -> List[str]:
        if final_states.batch_shape[0] != 0:
            a_arrays = np.vectorize(lambda x: self.aa_tokens[x] if x < 20 else "")(
                final_states.tensor[:,1:-1].cpu().numpy()
            )
        else:
            return []
        seqs = []

        for row in a_arrays:
            seqs.append("".join(row))
        return seqs
    
    @torch.inference_mode()  # is it legal?
    def seqs_to_trajectories(
        self, seqs: List[str], # module_pf: nn.Module
    ) -> Trajectories:
        """
        TODO: `Trajectories` should have a 'device' property,
        so that all tensors could be initialized on CPU (common practice for dataloader) and then transfer to something else.
        """
        # max_len=max([len(i) for i in seqs])
        seq_tensors, act_tensors, when_is_dones = [], [], []
        for seq in seqs:
            t = torch.full((self.max_length + 2, *self.state_shape), self.sf_code).long()
            a = torch.full(
                (self.max_length + 1, *self.action_shape), self.dummy_code
            ).long()
            l = len(seq)
            when_is_dones.append(l + 1)
            
            t[:,0]=self.s0_code
            for i, aa in enumerate(seq):
                aidx = self.aa_tokens.index(aa)
                t[i + 1 : l + 1, i+1] = aidx
                a[i] = aidx
            t[l + 1 :] = self.sf_code
            a[l] = self.exit_code
            seq_tensors.append(t)
            act_tensors.append(a)

        # states, actions, when_is_done
        # with torch.no_grad():
        states_tensor = torch.stack(seq_tensors, dim=1).to(self.device)
        action_tensor = torch.stack(act_tensors, dim=1).to(self.device)
        # states_class, action_class = self.make_states_class(), self.make_actions_class()
        states: DiscreteStates = self.States(states_tensor)
        actions: Actions = self.Actions(action_tensor)
        self.update_masks(states)
        when_is_done = torch.tensor(when_is_dones).to(self.device)

        # # log_probs

        # fw_probs = module_pf(states.tensor)
        # valid_state_mask = (states.tensor != self.sf_code).all(dim=-1)
        # # fw_probs[~states.forward_masks]=-torch.inf
        # fw_probs = torch.where(states.forward_masks, fw_probs, -torch.inf)  # -100.

        # fw_probs = torch.nn.functional.softmax(fw_probs, dim=-1)

        # # fw_probs[~valid_state_mask]=1

        # fw_probs = torch.where(
        #     valid_state_mask.unsqueeze(-1).repeat(
        #         *[1] * len(valid_state_mask.shape), fw_probs.shape[-1]
        #     ),
        #     fw_probs,
        #     1.0,
        # )

        # # a_tensor=actions.tensor.clone()
        
        # # a_tensor[a_tensor==self.dummy_code]=self.exit_code
        # a_tensor = torch.where(
        #     actions.tensor != self.dummy_code, actions.tensor, self.exit_code
        # )

        # log_probs = torch.log(
        #     torch.gather(input=fw_probs, dim=-1, index=a_tensor).squeeze(-1)
        # )
        
        # log_probs=torch.log(log_probs) .sum(dim=0)
        final_states = states[when_is_done - 1, torch.arange(len(seqs)).to(self.device)]
        
        log_rewards = self.log_reward(final_states)

        trajectories = Trajectories(
            env=self,
            states=states,
            conditioning=None,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=False,
            log_rewards=log_rewards,
            # log_probs=log_probs,
            log_probs=None,
            estimator_outputs=None,
        )

        return trajectories
    
    def trajectories_to_seqs(self,trajectories:Trajectories)->List[str]:
        batch_size=trajectories.states.batch_shape[-1]
        final_states=trajectories.states[trajectories.when_is_done-1,torch.arange(batch_size)]
        return self.states_to_seqs(final_states)

    def sample_random_seqs(self,n:int)->List[str]:
        o=[]
        for _ in range(n):
            l=random.randint(self.min_length,self.max_length)
            o.append(''.join(random.choices(self.aa_tokens,k=l)))
        return o
    ### offline sampling ###
    # TODO move to LightningDataModules
    # def collate_fn(self,seqs: List[str], module_pf: nn.Module) -> Trajectories:
    #     trajectories = self.seqs_to_trajectories(seqs, module_pf)
    #     return trajectories
    
    # def make_offline_dataloader(
    #     self, module_pf: nn.Module, 
    #     dataset:Dataset,
    #     **kwargs
    # ) -> DataLoader[Trajectories]:
    #     """
    #     kwargs for `DataLoader`
    #     """
    #     return DataLoader(
    #         dataset=dataset,
    #         collate_fn=partial(self.collate_fn, module_pf=module_pf),
    #         **kwargs,
    #     )
        
    ### Deprecated ###
    # def make_random_states_tensor(
    #     self, batch_shape: Tuple[int, ...]
    #     ) -> Tensor:
    #     '''
    #     Deprecated.
    #     Randomly sample some sequences, then `seqs_to_trajectories`
    #     '''
    #     states_tensor = torch.full(
    #         (*batch_shape, self.max_length), self.s0_code, device=self.device
    #     ).long()
    #     fill_until = torch.randint(0, self.max_length + 1, batch_shape)
    #     # fill_until=torch.full(batch_shape,self.max_length)
    #     for i in range(self.max_length):
    #         mask = i < fill_until
    #         random_numbers = torch.randint(
    #             0, self.max_length, batch_shape, device=self.device
    #         )
    #         states_tensor[(*mask.nonzero(as_tuple=True), i)] = random_numbers[mask]
    #     return states_tensor
        # states.set_default_typing()
        # # Not allowed to take any action beyond the environment height, but
        # # allow early termination.
        # states.set_nonexit_action_masks(
        #     states.tensor == self.height - 1,
        #     allow_exit=True,
        # )
        # states.backward_masks = states.tensor != 0
        
ADCPCycEnv=AdditivePepEnv