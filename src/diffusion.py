"Run diffusion model"

import os
import copy
from glob import glob
from functools import partial
from typing import List, Tuple, Callable 

import torch as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as L
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# internal
import src.positional_encoding as pe
from run.eval_flow import DictDataset

from tools.tools import misc, schedulers
from tools.tools.modules import IterativeNormLayer
from tools.tools.torch_utils import count_trainable_parameters, ema, make_ema
from tools.tools.lightning import get_loss


def push_to_device(sample, device):
    if isinstance(sample, dict):
        return {k: push_to_device(v, device) for k, v in sample.items()}
    elif isinstance(sample, T.Tensor):
        return sample.to(device)
    else:
        return sample

class TimeDependentAE(L.LightningModule):
    def __init__(self, network:partial, train_config:dict, eval_fw: Callable=None,  save_path=None, **kwargs):
        """
        should get an
        embedding network
        sampling network with solver
        evaluate framework
        """
        super().__init__()
        self.save_hyperparameters()

        self.ema_sampler_val = kwargs.get("ema_sampler_val")
        self.ema_embedder_val = kwargs.get("ema_embedder_val")
        self.train_config = train_config
        self.save_path=save_path
        self.eval_fw = eval_fw
        self.precision = kwargs.get("precision", 32)
        self.use_ema_in_eval = kwargs.get("use_ema_in_eval", True)
        self.ctp={}
        self.latn_dims = kwargs.get('latn_dims', 0)
        self.batch=None
        self.network = None
        self.ema_embedding_network=None
        self.ema_sampling_network=None
        self.idx = None
        self.ctp = {} # count trainable parameters
        
        # init loss function
        self.loss = get_loss(**self.train_config.get("loss_cfg", {'name':'mse'}))

        if not hasattr(network, 'forward'):
            self.sampling_network = network(loss=self.loss, latn_dims=self.latn_dims)
        else:
            self.sampling_network = network
            self.sampling_network.loss = self.loss
            
        if self.ema_sampler_val is not None:
            self.ema_sampling_network = make_ema(self.sampling_network)

        # count parameters
        self.ctp["Sampling size"] = count_trainable_parameters(self.sampling_network)
        self.init_val_log()

    def init_val_log(self):
        # setup validation step 
        self.validation_dict = {i: [] for i in ["ctxt", "gen_data", 
                                                "truth", "mask", "scalars"]}
        self.validation_dict['ctxt'] = {i: [] for i in ["mask", "scalars", 'cnts']}

    def configure_optimizers(self):
        "configure optimizer and scheduler. If scheduler is not in train_config, it will be None."

        # optimzer for sampling network and tasks (should be detached from embedding)
        opt_samp = {"optimizer": T.optim.AdamW(self.parameters(), **self.train_config["opt_cfg"])}

        # optimzer for embedding network
        if "lr_scheduler" in self.train_config:
            # config for scheduler if needed 
            opt_samp["lr_scheduler"] = schedulers.get_scheduler(optimizer=opt_samp["optimizer"],**self.train_config.lr_scheduler)
                                                                   
        return opt_samp

    def sampling_train_step(self, inpt:dict, training:bool=None) -> T.Tensor:
        training = self.training if training is None else training

        if (not training) and self.use_ema_in_eval:
            return self.ema_sampling_network.train_step(**inpt)
        else:
            return self.sampling_network.train_step(**inpt)
    
                
    def _shared_step(self, batch, batch_idx, log_name:str="train") -> Tuple[T.Tensor, T.Tensor, T.Tensor | None]:
        # total_loss = T.tensor(0.0, requires_grad=True)

        # get task loss on ema networks
        if 'ctxt' in batch:
            batch["ctxt"].pop("labels", None)
        
        # run sampling model
        diff_loss = self.sampling_train_step(batch)
        
        # log diffusion loss
        self.log(f"{log_name}/loss", diff_loss, prog_bar=True)
                
        return diff_loss

    def training_step(self, batch, batch_idx) -> T.Tensor:
        return self._shared_step(batch, batch_idx, log_name="train")
    
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        pass

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """update ema with new parameters"""
        with T.no_grad():
            if self.ema_embedder_val is not None:
                self.ema_embedding_network = ema(self.ema_embedding_network,
                                                 self.embedding_network,
                                                 self.ema_embedder_val)

            if self.ema_sampler_val is not None:
                self.ema_sampling_network = ema(self.ema_sampling_network,
                                                 self.sampling_network,
                                                 self.ema_sampler_val)
            
        if self.lr_schedulers is not None:
            if isinstance(self.lr_schedulers(), list):
                for i in self.lr_schedulers():
                    i.step()
            else:
                self.lr_schedulers().step() 

    def validation_step(self, batch:dict, batch_idx:int):
        "run validation batches and log results"
                
        # store input, mask and scalars for evaluation
        # log input data
        self.validation_dict["truth"].extend(batch["inpt"])
        self.validation_dict["mask"].extend(batch["mask"])
        for i in self.validation_dict['ctxt']:
            if i in batch["ctxt"]:
                self.validation_dict['ctxt'][i].extend(batch['ctxt'][i])

        # get val loss
        loss = self._shared_step(batch=copy.deepcopy(batch),
                                 batch_idx=batch_idx, log_name="valid")
        
        #generate samples
        # if len(self.validation_dict["generated"]):
        noise = T.randn_like(batch["inpt"])
        generated_sample = self.generate(noise=noise, ctxt=batch['ctxt'], mask=batch.get("mask", None))

        self.validation_dict["gen_data"].extend(T.nan_to_num(generated_sample["gen_data"], -999))

        return loss

    def on_validation_epoch_end(self):
        """log validation results over all valid batches"""
        for i,j in self.validation_dict.items():
            if len(j)==0:
                continue
            if isinstance(j, dict):
                for k,l in j.items():
                    if len(l)==0:
                        continue
                    vals = T.stack(l)
                    if vals.dtype == T.bool:
                        self.validation_dict[i][k] = vals.cpu().numpy()
                    else:
                        self.validation_dict[i][k] = vals.cpu().float().numpy()
            else:
                vals = T.stack(j)
                if vals.dtype == T.bool:
                    self.validation_dict[i] = vals.cpu().numpy()
                else:
                    self.validation_dict[i] = vals.cpu().float().numpy()
        
        if (self.eval_fw is not None):
            log_vals = self.eval_fw(**self.validation_dict)
    
            # log images
            if self.logger is not None:
                self.logger.experiment.log(log_vals, commit=False)
                
            # log additional values
            # for i,j in log_vals.items():
            #     self.log(f"valid/{i}", j, prog_bar=True)
        
        # # evaluate side task
        # for task in self.tasks:
        #     logs = task.get_eval()
        #     for name, vals in logs.items():
        #         self.log(f"valid/task/{task.name}_{name}", vals, prog_bar=True)

        # free memory
        self.init_val_log()
        # pass

    def generate_samples(self, initial_noise, disable_bar=True):

        generated_data={}
        
        # if isinstance(initial_noise, dict):
        #     initial_noise = DataLoader(DictDataset(initial_noise),
        #                                batch_size=256, shuffle=False)

        for sample in tqdm(initial_noise, disable=disable_bar # len(self.initial_noise)==1
                            ):
            # sample = {i:j.to(self.device) for i,j in sample.items()}
            
            # if 'noise' not in sample:
            #     cnts = sample.pop('cnts')
            # if 'scalar' not in sample:
            #     sample['ctxt'] = sample.pop('scalars')

            # sample['noise'] = T.randn(cnts.shape).to(cnts)
            sample['noise'] = sample.pop('inpt')
            
            sample = push_to_device(sample, self.device)

            _generated = self.generate(**sample)

            _generated = push_to_device(_generated, 'cpu')
            
            # concat to generated_data
            for i,j in _generated.items():
                if isinstance(j, dict): # for dict nested ctxt
                    if i not in generated_data:
                        generated_data[i]={}
                    for k,l in j.items():
                        if k not in generated_data[i]:
                            generated_data[i][k]=T.tensor([])
                        
                        generated_data[i][k] = T.concat([generated_data[i][k], l],0)
                else:
                    if T.isnan(_generated[i]).any():
                        print("why")
                    if i not in generated_data:
                        generated_data[i]=T.tensor([])
                    generated_data[i] = T.concat([generated_data[i], _generated[i]],0)

        return generated_data

    def generate(self, noise,  ctxt=None, mask=None, n_steps:int=None):
        if self.ema_sampler_val is not None and self.use_ema_in_eval:
            return self.ema_sampling_network.generate(noise=noise, ctxt=ctxt, mask=mask,
                                                      n_steps=n_steps)
        else:
            return self.sampling_network.generate(noise=noise, ctxt=ctxt, mask=mask,
                                                  n_steps=n_steps)


class Classifier(L.LightningModule):
    def __init__(self, network:partial, train_config:dict, eval_fw: Callable=None,  save_path=None, **kwargs):
        """
        should get an
        embedding network
        sampling network with solver
        evaluate framework
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_config = train_config
        self.save_path=save_path
        self.eval_fw = eval_fw
        self.precision = kwargs.get("precision", 32)
        self.ctp={}
        
        # init loss function
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=T.tensor(5))
        

        self.network = network
            
        # count parameters
        self.ctp["Classifier size"] = count_trainable_parameters(self.network)
        self.init_val_log()

    def init_val_log(self):
        # setup validation step 
        self.validation_dict = {i: [] for i in ["ctxt", "gen_data", 
                                                "truth", "mask", "scalars"]}
        self.validation_dict['ctxt'] = {i: [] for i in ["mask", "scalars", 'cnts']}

    def configure_optimizers(self):
        "configure optimizer and scheduler. If scheduler is not in train_config, it will be None."

        # optimzer for sampling network and tasks (should be detached from embedding)
        opt_samp = {"optimizer": T.optim.AdamW(self.parameters(), **self.train_config["opt_cfg"])}

        # optimzer for embedding network
        if "lr_scheduler" in self.train_config:
            # config for scheduler if needed 
            opt_samp["lr_scheduler"] = schedulers.get_scheduler(optimizer=opt_samp["optimizer"],**self.train_config.lr_scheduler)
                                                                   
        return opt_samp
                
    def _shared_step(self, batch, batch_idx, log_name:str="train") -> Tuple[T.Tensor, T.Tensor, T.Tensor | None]:

        # get task loss on ema networks
        if 'ctxt' in batch:
            batch["ctxt"].pop("labels", None)
        
        # run sampling model
        ctxt = batch.pop('ctxt')
        output = self.network(**ctxt)
        
        labels = T.concat(
            [
                batch['mask']*1.0,
                T.zeros((len(batch['inpt']), ctxt['mask'].size()[-1]-batch['mask'].size()[-1]),device=self.device)],1
            )

        # loss = self.loss(output[ctxt['mask']], labels[ctxt['mask']])
        loss = T.nn.functional.binary_cross_entropy_with_logits(output, labels, weight=ctxt['mask']*1)
        
        acc = T.sum(
            (output[ctxt['mask']]>0) == labels[ctxt['mask']].bool()
            )/ctxt['mask'].sum()
        
        # calculate AUC
        auc = roc_auc_score(
            labels[ctxt['mask']].cpu().detach().numpy(), 
            T.sigmoid(output[ctxt['mask']]).cpu().detach().numpy())
        
        # log diffusion loss
        self.log(f"{log_name}/loss", loss, prog_bar=True)

        self.log(f"{log_name}/accuracy", acc, prog_bar=True)

        self.log(f"{log_name}/auc", auc, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx) -> T.Tensor:
        return self._shared_step(batch, batch_idx, log_name="train")
    
    def on_train_epoch_start(self, *args, **kwargs) -> None:
        pass

    def on_train_batch_end(self, *args, **kwargs) -> None:
        """update ema with new parameters"""
        if self.lr_schedulers is not None:
            if isinstance(self.lr_schedulers(), list):
                for i in self.lr_schedulers():
                    i.step()
            else:
                self.lr_schedulers().step() 

    def validation_step(self, batch:dict, batch_idx:int):
        "run validation batches and log results"
                
        # get val loss
        loss = self._shared_step(batch=copy.deepcopy(batch),
                                 batch_idx=batch_idx, log_name="valid")

        return loss

    def on_validation_epoch_end(self):
        """log validation results over all valid batches"""
        # free memory
        self.init_val_log()
        