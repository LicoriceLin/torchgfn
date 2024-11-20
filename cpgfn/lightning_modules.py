# %%
import lightning as L
from typing import Literal,List,Any,Dict
from .envs import AdditivePepEnv
from .models import BaseBackboneModule,CircularEncoder,OneHotBackbone
from gfn.env import Env,DiscreteEnv
from gfn.modules import DiscretePolicyEstimator
from lightning.pytorch.cli import LightningCLI
from torch import Tensor,nn
import torch
from gfn.gflownet import TBGFlowNet,FMGFlowNet
from gfn.containers import Trajectories
from .utils.plots import aa_dist,seqlength_dist,reward_dist,simple_aagroup_dist
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import IterableDataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR
from math import log
from lightning.pytorch.callbacks import BaseFinetuning,BasePredictionWriter
from torch.optim.optimizer import Optimizer
from typing_extensions import override
from functools import partial
'''
how to check params: 
python -m cpgfn.lightning_modules fit --model.env.help AdditivePepEnv
'''


# %%
class AdditivePepTBModule(L.LightningModule):
    logger:TensorBoardLogger
    gfn:TBGFlowNet|FMGFlowNet
    '''
    #TODO
    try to group the parameters?
    '''
    def __init__(self,   
                 
        env:AdditivePepEnv,
        backbone:BaseBackboneModule,
        # modes selection
        loss_mode:Literal['tb','fm','fit']='tb',
        sample_mode:Literal['gfn','random']='gfn',
        # head properties
        head_hidden_dim:int=512,
        final_norm:bool=False,
        head_num_layers:int=1,
        # batch sizes
        train_bs:int=32,
        val_bs:int=512,
        # learning rates
        lr:float=0.1,
        logZ_lr_coef:float=1e3, # adjust logZ0 lr
        denom_lr:float=10, # adjust backbone lr (1/ratio to head_lr)
        warmup_lr_coef:float=1e-2,
        # iteration periods 
        val_check_interval:int=500, # overide trainers in callbacks
        scheduler_every_val:int=2, # every x validation, update the scheduler 
        plot_every_val:int=20, # every x validation, draw the plots
        freeze_backbone_val:int=0, # after x validation, unfreeze backbone
        warm_up_iter:int=2, 
            # `fwd_scheduler_lambda`'s f_step & w_step. 
            # so the global warm_up step = scheduler_every_val * scheduler_every_val * warm_up_iter
            # GFN loss property
        z0:float=0.,
        alpha: float = 1.0,
        # sampling bias
        sf_bias: float = 0.0,
        epsilon: float = 0.0,
        temperature: float = 1.0,
        # param loaders for: 
            # 1) load parameters for finetune; 
            # 2) load legacy checkpoints 
        load_mode:Literal['full','backbone']='full',
        load_state_from:str|None = None,
        ):
        super().__init__()
        self.automatic_optimization = False
        self.env=env.to(self.device)

        # modes selection
        self.loss_mode=loss_mode
        self.sample_mode=sample_mode
        
        # head properties
        self.head_hidden_dim=head_hidden_dim
        self.head_num_layers=head_num_layers
        self.final_norm=final_norm

        # batch sizes
        self.train_bs=train_bs
        self.val_bs=val_bs

        # learning rates
        self.lr=lr
        self.logZ_lr_coef=logZ_lr_coef
        self.denom_lr=denom_lr
        self.warmup_lr_coef=warmup_lr_coef
        # iteration periods 
        self.val_check_interval=val_check_interval 
        self.scheduler_every_val=scheduler_every_val
        self.plot_every_val=plot_every_val
        self.freeze_backbone_val=freeze_backbone_val
        self.warm_up_iter=warm_up_iter

        #sampling bias
        self.sf_bias = sf_bias
        self.epsilon = epsilon
        self.temperature = temperature

        #temperal,for logging/plots
        self.max_length=self.env.max_length
        self.beta=self.env.reward_kwargs.get('beta',0.)
        self.max_score=sum([v for k,v in self.env.reward_kwargs.items() if 'score' in k])
        self.log_s=self.env.reward_kwargs.get('log_s',False)
        
        if self.loss_mode=='tb':   
            pf_estimator = DiscretePolicyEstimator(
                add_head(backbone,head_hidden_dim,outputs_dim=env.n_actions,
                        norm=final_norm,num_layers=head_num_layers),
                n_actions=env.n_actions,
                is_backward=False,
                preprocessor=env.preprocessor,
            )

            pb_estimator = DiscretePolicyEstimator(
                add_head(backbone,head_hidden_dim,outputs_dim=env.n_actions-1,
                         norm=final_norm,num_layers=head_num_layers),
                n_actions=env.n_actions,
                is_backward=True,
                preprocessor=env.preprocessor,
            )
            self.gfn = TBGFlowNet(logZ=z0, pf=pf_estimator, pb=pb_estimator)
            
        elif self.loss_mode=='fm':
            logf_estimator=DiscretePolicyEstimator(
                add_head(backbone,head_hidden_dim,outputs_dim=env.n_actions,
                         norm=final_norm,num_layers=head_num_layers),
                n_actions=env.n_actions,
                is_backward=False,
                preprocessor=env.preprocessor,
            )
            self.gfn=FMGFlowNet(logF=logf_estimator,alpha=alpha)
            
        elif self.loss_mode=='fit':
            assert self.sample_mode=='random'
            self.reward_estimator=add_head(backbone,head_hidden_dim,
                outputs_dim=1,norm=final_norm,num_layers=head_num_layers)
            self.loss=nn.MSELoss()
        else:
            raise ValueError
        
        if load_state_from is not None:
            state_dict:Dict[str,Tensor]=torch.load(load_state_from)['state_dict']
            if load_mode=='full':
                self.load_state_dict(state_dict)
            elif load_mode=='backbone':
                # now only consider partial load from fit module 
                state_dict={k.replace('reward_estimator.0.',''):v 
                    for k,v in state_dict.items() if 'reward_estimator.0.' in k}
                if self.loss_mode=='tb':
                    self.gfn.pf.module[0].load_state_dict(state_dict)
                elif self.loss_mode=='fm':
                    self.gfn.logF.module[0].load_state_dict(state_dict)
                elif self.loss_mode=='fit':
                    self.reward_estimator[0].load_state_dict(state_dict)
                else:
                    raise ValueError
            else:
                raise ValueError
            
    def training_step(self, batch:Tensor, batch_idx):
        # with torch.no_grad():
        if self.sample_mode=='gfn':
            trajectories=self.gfn.sample_trajectories(self.env,self.train_bs,
                sf_bias=self.sf_bias, epsilon=self.epsilon,temperature=self.temperature)
        elif self.sample_mode=='random':
            trajectories=self.env.seqs_to_trajectories(
                self.env.sample_random_seqs(self.train_bs)
            )
        else:
            raise ValueError
        
        r=trajectories.log_rewards-self.beta
        if self.log_s:
            r=torch.exp(r)
        
        # import pdb;pdb.set_trace()
        if self.loss_mode=='tb':
            loss=self.gfn.loss(self.env, trajectories=trajectories)
        elif self.loss_mode=='fm':
            samples=self.gfn.to_training_samples(trajectories)
            loss=self.gfn.loss(self.env,states_tuple=samples)
        elif self.loss_mode=='fit':
            final_states=trajectories.states[trajectories.when_is_done - 1, 
                    torch.arange(trajectories.when_is_done.shape[0]
                    ).to(trajectories.when_is_done.device)]
            loss=self.loss(
                self.reward_estimator(
                    self.env.preprocessor(final_states).clone()).reshape(-1),
                r/self.max_score-0.5)
        else:
            raise ValueError
        self.log('train/loss',loss)
        if self.loss_mode!='fit':
            self.log('train/mean_rewards',r.mean())
            self.log('train/mean_len',trajectories.when_is_done.float().mean()-1)
        # return loss

        self.manual_backward(loss)
        for optimizer in self.optimizers():
            optimizer:torch.optim.Optimizer
            optimizer.step()
            optimizer.zero_grad()

        true_step=self.trainer.global_step/len(self.optimizers())
        sch_interval=self.val_check_interval*self.scheduler_every_val
        if (true_step) % (sch_interval) ==0 and true_step>sch_interval:
            schs=self.lr_schedulers()
            if schs is None:
                return
            else: 
                if not isinstance(schs,list):
                    schs=[schs]
                for sch in schs:
                    sch:torch.optim.lr_scheduler.LRScheduler
                    if isinstance(sch,ReduceLROnPlateau):
                        sch.step(val_loss=self.trainer.callback_metrics["val_loss"])
                    else:
                        sch.step()
    
    def validation_step(self, batch:Tensor, batch_idx):
        # if self.sample_mode=='gfn':
        if self.loss_mode in ['tb','fm']:
            trajectories=self.gfn.sample_trajectories(self.env,self.val_bs)
        # elif self.sample_mode=='random':
        elif self.loss_mode == 'fit':
            trajectories=self.env.seqs_to_trajectories(
                self.env.sample_random_seqs(self.val_bs)
            )
        else:
            raise ValueError
        
        final_states=trajectories.states[trajectories.when_is_done - 1, 
                    torch.arange(trajectories.when_is_done.shape[0]
                    ).to(trajectories.when_is_done.device)]
        
        r=trajectories.log_rewards-self.beta
        if self.log_s:
            r=torch.exp(r)
                
        if self.loss_mode in ['tb','fm']:
            
            
            seqs=self.env.states_to_seqs(final_states)
            
            if self.loss_mode=='tb':
                loss=self.gfn.loss(self.env, trajectories=trajectories)
            elif self.loss_mode=='fm':
                samples=self.gfn.to_training_samples(trajectories)
                loss=self.gfn.loss(self.env,states_tuple=samples)
            else:
                raise ValueError
            
            #TODO multiple GPU training compatible
            writer:SummaryWriter=self.logger._experiment
            # if isinstance(self.optimizers(),list):
            #     true_step=self.trainer.global_step/len(self.optimizers())
            # else:
            #     true_step=self.trainer.global_step
            true_step=self.trainer.global_step//self.n_optimizers
            if true_step%(self.val_check_interval*self.plot_every_val)==0:
                writer.add_figure(tag='val/length_dist',
                        figure=seqlength_dist(seqs,max_length=self.max_length)[0],global_step=true_step)
                writer.add_figure(tag='val/aa_dist',
                        figure=simple_aagroup_dist(seqs,max_length=self.max_length)[0],global_step=true_step)
                
                writer.add_figure(tag='val/reward_dist',
                        figure=reward_dist(r,max_score=self.max_score)[0],global_step=true_step)
                
                writer.add_text('val/sample_seq','\n'.join(seqs[:5]),global_step=true_step)
            writer.add_scalar("val/mean_rewards",torch.exp(trajectories.log_rewards-self.beta).mean(),global_step=true_step)
            writer.add_scalar("val/mean_log_rewards",trajectories.log_rewards.mean()-self.beta,global_step=true_step)
            writer.add_scalar("val/loss", loss.item(),global_step=true_step)
            writer.add_scalar('val/mean_len',trajectories.when_is_done.float().mean()-1,global_step=true_step)
            if self.loss_mode=='tb':
                writer.add_scalar('val/z0',self.gfn.logZ.item(),global_step=true_step)
            # elif self.loss=='fm':
            
        elif self.loss_mode == 'fit':
            loss=self.loss(
                self.reward_estimator(
                    self.env.preprocessor(final_states)).reshape(-1),
                r/self.max_score-0.5 #0.5: drag mean to 0.
                )
        else:
            raise ValueError
        
        self.log("val_loss", loss.item())
        
    def predict_step(self, batch:Tensor, batch_idx):
        '''
        #TODO BasePredictionWriters
        '''
        if self.loss_mode == 'fit':
            seqs=self.env.sample_random_seqs(self.val_bs)
            trajectories=self.env.seqs_to_trajectories(seqs)
            
            final_states=trajectories.states[trajectories.when_is_done - 1, 
                    torch.arange(trajectories.when_is_done.shape[0]
                    ).to(trajectories.when_is_done.device)]
            
            r=trajectories.log_rewards-self.beta
            if self.log_s:
                r=torch.exp(r)
            
            pred_r=(self.reward_estimator(
                    self.env.preprocessor(final_states).clone()).reshape(-1)+0.5)*self.max_score
            ret=dict(seqs=seqs,r=r,pred_r=pred_r)
            torch.save(ret,'predict.pt')
            return ret
        else:
            raise NotImplementedError
        
    # def on_predict_epoch_end(self):
    #     super().on_predict_epoch_end()
        # return super().predict_step(*args, **kwargs)
    def configure_optimizers(self):
        if self.loss_mode=='tb':
            core=self.gfn.pf.module
        elif self.loss_mode=='fm':
            core=self.gfn.logF.module
        elif self.loss_mode=='fit':
            core=self.reward_estimator
        else:
            raise NotImplementedError
        core:nn.Sequential

        if self.freeze_backbone_val==0:
            basic_params=core.parameters()
        else:
            basic_params=core[1:].parameters()
        
        optimizer = torch.optim.Adam(basic_params, lr=self.lr)
        frequency=self.val_check_interval*self.scheduler_every_val
        if self.loss_mode=='tb':
            optimizer_z=torch.optim.SGD(params= self.gfn.logz_parameters()
                    ,lr=self.lr*self.logZ_lr_coef)
            scheduler=LambdaLR(optimizer,lr_lambda=partial(fwd_scheduler_lambda,
                    f_step=self.warm_up_iter,w_step=self.warm_up_iter,
                    decay_after=0.998,norm_init=self.warmup_lr_coef))
            scheduler_z=LambdaLR(optimizer_z,
                lr_lambda=partial(fwd_scheduler_lambda,
                    f_step=self.warm_up_iter//2,w_step=self.warm_up_iter//2,
                    decay_after=0.9998,norm_init=self.warmup_lr_coef/10))
            # scheduler_lambdas=[]
            return ([optimizer,optimizer_z],
                    [
                {"scheduler": scheduler,
                 "monitor": "val_loss",
                 "interval": "step",
                 "frequency": frequency,
                 "strict":False
                 },
                  {"scheduler": scheduler_z,
                 "monitor": "val_loss",
                 "interval": "step",
                 "frequency": frequency,
                 "strict":False
                 } ])
            # optimizer.add_param_group({"params": self.gfn.logz_parameters(), 
            #     "lr": self.lr*self.logZ_lr_coef})
            # scheduler_lambdas=[scheduler_lambda,lambda epoch: 0.999**epoch]
        elif self.loss_mode == 'fm':
            scheduler=LambdaLR(optimizer,lr_lambda=partial(fwd_scheduler_lambda,
                    f_step=self.warm_up_iter,w_step=self.warm_up_iter,
                    decay_after=0.998,norm_init=self.warmup_lr_coef))
            return {
            "optimizer": optimizer, 
            "lr_scheduler": 
                {"scheduler": scheduler,
                 "monitor": "val_loss",
                 "interval": "step",
                 "frequency": frequency,
                 "strict":False
                 } 
            }
        elif self.loss_mode == 'fit':
            # def scheduler_lambda_fit(epoch:int):
            #     if epoch<=100:
            #         return 1.
            #     else:
            #         return 0.99**(epoch-100)
            scheduler=LambdaLR(optimizer,lr_lambda=partial(
                fwd_scheduler_lambda,f_step=10,w_step=10,decay_after=0.99,norm_init=1.0))
            return {
                "optimizer": optimizer, 
                "lr_scheduler": 
                    {"scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "step",
                    "frequency": frequency,
                    "strict":False
                    } 
                }
        else:
            raise NotImplementedError
        
        

        # if self.loss_mode =='tb':
        #     optimizer = torch.optim.Adam(self.gfn.pf_pb_parameters(), lr=self.lr)
        #     optimizer.add_param_group({"params": self.gfn.logz_parameters(), "lr": self.lr*1e3})
        #     scheduler=LambdaLR(optimizer,lr_lambda=[scheduler_lambda,lambda epoch: 0.98**epoch])
        # elif self.loss_mode == 'fm':
        #     optimizer = torch.optim.Adam(self.gfn.parameters(), lr=self.lr)
        #     scheduler=LambdaLR(optimizer,lr_lambda=scheduler_lambda)
        # elif self.loss_mode == 'fit':
        #     def scheduler_lambda(epoch:int):
        #         if epoch<=100:
        #             return 1.
        #         else:
        #             return 0.99**(epoch-100)
        #     optimizer = torch.optim.Adam(self.reward_estimator.parameters(), lr=self.lr)
        #     scheduler=LambdaLR(optimizer,lr_lambda=scheduler_lambda)
        # else:
        #     raise ValueError
        # scheduler=ReduceLROnPlateau(optimizer,threshold=1e-4,min_lr=self.lr*1e-4,patience=20,factor=0.5)

        

        # return [optimizer],[scheduler]
    # def 
    
    def configure_callbacks(self):
        # TODO configure predict writer
        if self.freeze_backbone_val>0:
            return [
                BackboneFreeze(
                    unfreeze_at_step=self.freeze_backbone_val*self.val_check_interval,
                    denom_lr=self.denom_lr)
                    ]
        
    def to(self,*args: Any, **kwargs: Any):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        self.env.to(device=device,dtype=dtype)
        return super().to(*args, **kwargs)
        
    @property
    def n_optimizers(self)->int:
        # print(f"ffff {self.optimizers()}")
        # import pdb;pdb.set_trace()
        if getattr(self,'_n_optimizers',0)==0:
            optimizers=self.optimizers()
            if isinstance(optimizers,list) and len(optimizers)>0:
                # print(f"ffff {len(optimizers)}")
                self._n_optimizers=len(optimizers)
            elif isinstance(optimizers,torch.optim.Optimizer):
                self._n_optimizers=1
            else:
                Warning('.optimizers() not initialized! Return 1 as placeholder')
                return 1
        return self._n_optimizers
    
def add_head(backbone:BaseBackboneModule,
            head_hidden_dim:int,
            outputs_dim:int,
            dropout:float=0.1,
            num_layers:int=1,
            norm:bool=True,
            elementwise_affine:bool=True,
            bias:bool=True
            )->nn.Sequential:
    '''
    TODO leave backbone outside of the `nn.Sequential`
    '''
    modules:List[nn.Module]=[backbone,nn.Linear(backbone.embedding_dim, head_hidden_dim)]
    for _ in range(num_layers-1):
        modules.extend(
            [
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=head_hidden_dim),
            nn.Linear(head_hidden_dim, head_hidden_dim)
            ])
    modules.extend([nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=head_hidden_dim),
            nn.Linear(head_hidden_dim,outputs_dim)
            ])
    
    if norm:
        if outputs_dim>1:
            modules.append(nn.LayerNorm(normalized_shape=outputs_dim,
                elementwise_affine=elementwise_affine,bias=bias))
        else:
            modules.append(nn.Sigmoid())
    
    return nn.Sequential(*modules)

def fwd_scheduler_lambda(epoch:int,f_step=50,w_step=50,norm_init=0.01,decay_after=0.998):
    '''
    freeze, warmup and decay scheduler lambda
    '''
    # TODO go to init parameters
    i=(1/norm_init)**(1/(w_step))
    # print(i)
    if epoch<=f_step:
        return norm_init
    elif epoch<=(w_step+f_step):
        return norm_init* (i**(epoch-f_step))
    else:
        return decay_after**(epoch-w_step-f_step)
    
# %%
class BackboneFreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_step=5000,denom_lr:float=10):
        super().__init__()
        self._unfreeze_at_step = unfreeze_at_step
        self.denom_lr=denom_lr

    def freeze_before_training(self, pl_module: "AdditivePepTBModule") -> None:
        """Override to add your freeze logic."""
        
        if pl_module.loss_mode=='tb':
            backbone=pl_module.gfn.pf.module[0]
        elif pl_module.loss_mode=='fm':
            backbone=pl_module.gfn.logF.module[0]
        elif pl_module.loss_mode=='fit':
            backbone=pl_module.reward_estimator[0]
        else:
            raise NotImplementedError
        self.freeze(backbone)
    
    def finetune_function(self, pl_module: "AdditivePepTBModule", current_step: int, optimizer: Optimizer) -> None:
        """forced align to the basic params group."""
        true_step=current_step//pl_module.n_optimizers
        if true_step == self._unfreeze_at_step:
            if pl_module.loss_mode=='tb':
                backbone=pl_module.gfn.pf.module[0]
            elif pl_module.loss_mode=='fm':
                backbone=pl_module.gfn.logF.module[0]
            elif pl_module.loss_mode=='fit':
                backbone=pl_module.reward_estimator[0]
            else:
                raise NotImplementedError
            
            current_lr = optimizer.param_groups[0]["lr"]
            self.unfreeze_and_add_param_group(
                modules=backbone,
                optimizer=optimizer,
                train_bn=True,
                lr=current_lr/self.denom_lr
                )
            
        # 
        elif true_step > self._unfreeze_at_step:
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[-1]["lr"] = current_lr/self.denom_lr

        
        
    @override
    def on_train_batch_start(self,
        trainer: "L.Trainer", pl_module: "AdditivePepTBModule", batch: Any, batch_idx: int
        ) -> None:
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            # TODO better logic to ignore unwanted optimizers.
            num_param_groups = len(optimizer.param_groups)
            if isinstance(optimizer,torch.optim.Adam):
                self.finetune_function(pl_module, trainer.global_step, optimizer)
            current_param_groups = optimizer.param_groups
            self._store(pl_module, opt_idx, num_param_groups, current_param_groups)
                
                # only add to the `Adam`


    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: "AdditivePepTBModule") -> None:
        return
    
    @override
    def setup(self, trainer: L.Trainer, pl_module: "AdditivePepTBModule",stage: str) -> None:
        #TODO another utils callback
        trainer.val_check_interval=pl_module.val_check_interval

        optimizers=pl_module.optimizers()
        if isinstance(optimizers,list) and len(optimizers)>=1:
            pl_module._n_optimizers=len(optimizers)
            Warning(f'_n_optimizers = { pl_module._n_optimizers}')
        elif isinstance(optimizers,torch.optim.Optimizer):
            pl_module._n_optimizers=1
            Warning(f'_n_optimizers = { pl_module._n_optimizers}')
        else:
            Warning('.optimizers() not initialized! Return 1 as placeholder')
            pl_module._n_optimizers=0
        
        Warning(f'n_optimizers = {pl_module.n_optimizers}')
        trainer.fit_loop.epoch_loop.max_steps=trainer.max_steps*pl_module.n_optimizers

        return super().setup(trainer, pl_module,stage)

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
    
    def predict_dataloader(self):
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
    
    def predict_dataloader(self):
        return self._dataloader(bs=self.val_bs)
    
    
    
# %% 


if __name__=='__main__':
    cli=LightningCLI(model_class=AdditivePepTBModule,
        datamodule_class=PseudoDataModule,
        parser_kwargs={"parser_mode": "omegaconf"})
# LightningCLI
# 
# cli.config