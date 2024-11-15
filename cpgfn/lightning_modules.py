# %%
import lightning as L
from typing import Literal,List,Any
from .envs import AdditivePepEnv
from .models import BaseBackboneModule,CircularEncoder,OneHotBackbone
from gfn.env import Env,DiscreteEnv
from gfn.modules import DiscretePolicyEstimator
from lightning.pytorch.cli import LightningCLI
from torch import Tensor,nn
import torch
from gfn.gflownet import TBGFlowNet
from gfn.containers import Trajectories
from .utils.plots import aa_dist,seqlength_dist,reward_dist
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import IterableDataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
'''
how to check params: 
python -m cpgfn.lightning_modules fit --model.env.help AdditivePepEnv
'''


# %%
class AdditivePepTBModule(L.LightningModule):
    logger:TensorBoardLogger
    
    def __init__(self,   
        env:AdditivePepEnv,
        backbone:BaseBackboneModule,
        head_hidden_dim:int=512,
        train_bs:int=32,
        val_bs:int=512,
        lr:float=0.1,
        z0:float=0.,
        load_state_from:str|None = None,
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        # *args, 
        # **kwargs
        ):
        super().__init__()
        self.env=env.to(self.device)
        self.head_hidden_dim=head_hidden_dim
        self.train_bs=train_bs
        self.val_bs=val_bs
        self.lr=lr
        self.sf_bias=sf_bias
        self.epsilon = epsilon
        self.temperature=temperature
        pf_estimator = DiscretePolicyEstimator(
            add_head(backbone,head_hidden_dim,env.n_actions),
            n_actions=env.n_actions,
            is_backward=False,
            preprocessor=env.preprocessor,
        )

        pb_estimator = DiscretePolicyEstimator(
            add_head(backbone,head_hidden_dim,env.n_actions-1),
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
        )
        self.gfn = TBGFlowNet(logZ=z0, pf=pf_estimator, pb=pb_estimator)
        if load_state_from:
            self.load_state_dict(torch.load(load_state_from)['state_dict'])
    def training_step(self, batch:Tensor, batch_idx):
        # with torch.no_grad():
        trajectories=self.gfn.sample_trajectories(self.env,self.train_bs,
            sf_bias=self.sf_bias, epsilon=self.epsilon,temperature=self.temperature)
            
        # import pdb;pdb.set_trace()
        loss=self.gfn.loss(self.env, trajectories)
        self.log('train/loss',loss)
        self.log('train/mean_rewards',torch.exp(trajectories.log_rewards-1).mean())
        self.log('train/mean_len',trajectories.when_is_done.float().mean()-1)
        return loss
    def validation_step(self, batch:Tensor, batch_idx):
        trajectories=self.gfn.sample_trajectories(self.env,self.val_bs)
        
        final_states=trajectories.states[trajectories.when_is_done - 1, 
                torch.arange(trajectories.when_is_done.shape[0]
                ).to(trajectories.when_is_done.device)]
        
        seqs=self.env.states_to_seqs(final_states)
        print(seqs[:5])
        
        loss = self.gfn.loss(self.env, trajectories)
        
        global_step=self.trainer.global_step
        
        
        
        
        #TODO multiple GPU training compatible
        # self.logger.experiment:SummaryWriter
        writer:SummaryWriter=self.logger._experiment
        writer.add_figure(tag='val/length_dist',
                figure=seqlength_dist(seqs,max_length=self.env.max_length)[0],global_step=global_step)
        writer.add_figure(tag='val/aa_dist',
                figure=aa_dist(seqs,max_length=self.env.max_length)[0],global_step=global_step)
        #TODO -1 -> beta, 20 -> max_score
        writer.add_figure(tag='val/reward_dist',
                figure=reward_dist(torch.exp(trajectories.log_rewards-1.),max_score=20)[0],global_step=global_step)
        
        self.log("val/mean_rewards",torch.exp(trajectories.log_rewards-1.).mean())
        self.log("val/loss", loss.item())
        self.log('val/z0',self.gfn.logZ.item())
        self.log("val_loss", loss.item())
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gfn.pf_pb_parameters(), lr=self.lr)
        optimizer.add_param_group({"params": self.gfn.logz_parameters(), "lr": self.lr*10})
        scheduler=ReduceLROnPlateau(optimizer,threshold=0.05,min_lr=self.lr*1e-4,patience=5,factor=0.5)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": 
                {"scheduler": scheduler,
                 "monitor": "val_loss",
                 "interval": "step",
                 "frequency": 100,
                 "strict":False
                 } 
            }
        # return [optimizer],[scheduler]
    # def 
    
    def to(self,*args: Any, **kwargs: Any):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        self.env.to(device=device,dtype=dtype)
        return super().to(*args, **kwargs)
        
        
def add_head(backbone:BaseBackboneModule,
            head_hidden_dim:int,
            n_actions:int,
            dropout:float=0.1)->nn.Module:
    return nn.Sequential(backbone,
            nn.Linear(backbone.embedding_dim, head_hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(head_hidden_dim,n_actions),
            nn.LayerNorm(normalized_shape=n_actions)
            )

# %%
class GFNSampler(IterableDataset):
    def __init__(self,module:AdditivePepTBModule,
                 bs:int=32):
        self.module=module
        self.bs=bs
        super().__init__()
        
    @torch.inference_mode()
    def __iter__(self):
        m=self.module
        while True:
            # trajectories=
            # print(m.env.device)
            # print(m.device)
            traj=m.gfn.sample_trajectories(m.env,self.bs)
            # print(traj.states.tensor.device)
            # import pdb;pdb.set_trace()
            yield traj
        # return super().__iter__()
        
class GFNSamplerDataModule(L.LightningDataModule):
    def __init__(self,
            # module:AdditivePepTBModule,
            train_bs:int=32,
            val_bs:int=512):
        super().__init__()
        # self.module=module
        self.train_bs=train_bs
        self.val_bs=val_bs
        
    def setup(self, stage):
        # super().setup()
        self.stage=stage
        assert self.trainer is not None
        self.module=self.trainer.lightning_module
        # import pdb;pdb.set_trace()
        
    def collate_fn(self,items:List[Trajectories])->Trajectories:
        assert len(items)==1
        return items[0]
    
    def _dataloader(self,bs:int):
        return DataLoader(GFNSampler(module=self.module,bs=bs),
            batch_size=1,num_workers=1,collate_fn=self.collate_fn)
        
    def val_dataloader(self):
        return self._dataloader(bs=self.val_bs)
    
    def train_dataloader(self):
        return self._dataloader(bs=self.train_bs)
        # return super().train_dataloader()
    
    def transfer_batch_to_device(self, batch:Trajectories, device, dataloader_idx)->Trajectories:
        # ['env', , 'is_backward', 'states', 'actions', ,]
        for t in [batch,batch.states,batch.actions]:
            for k,v in tuple(t.__dict__.items()):
                if isinstance(v,Tensor):
                    setattr(t,k,v.to(device))
        return batch
    
        # super().transfer_batch_to_device(batch, device, dataloader_idx)
    
class PsudoDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        # self.bs=bs

    def __iter__(self):
        while True:
            yield torch.zeros(1)
            # trajectories=
            # print(m.env.device)
            # print(m.device)
            # traj=m.gfn.sample_trajectories(m.env,self.bs)
            
            
class PseudoDataModule(L.LightningDataModule):
    def __init__(self,
            # module:AdditivePepTBModule,
            train_bs:int=32,
            val_bs:int=512,
            max_worker:int=16):
        self.train_bs=train_bs
        self.val_bs=val_bs
        super().__init__()
        
    def setup(self, stage):
        return super().setup(stage)
    
    def _dataloader(self,bs:int):
        return DataLoader(PsudoDataset(),
            batch_size=bs,num_workers=16)
        
    def val_dataloader(self):
        return self._dataloader(bs=self.val_bs)
    
    def train_dataloader(self):
        return self._dataloader(bs=self.train_bs)
    
    
    
# %% 


if __name__=='__main__':
    cli=LightningCLI(model_class=AdditivePepTBModule,
        datamodule_class=PseudoDataModule,
        parser_kwargs={"parser_mode": "omegaconf"})
# LightningCLI
# 
# cli.config