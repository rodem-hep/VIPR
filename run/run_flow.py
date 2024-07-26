"train flow to estimate p(N)"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import hydra
import numpy as np
import torch as T

from src.eval_utils import EvaluateFramework, get_percentile

from tools.flows import Flow, stacked_norm_flow
from tools.modules import IterativeNormLayer, MinMaxLayer
from tools.omegaconf_utils import instantiate_collection
from tools import misc

def load_flow_from_path(path:str, dev:str="cuda"):
    "load flow from path and return it"
    flow_cfg = misc.load_yaml(f"{path}/.hydra/config.yaml")
    # flow = hydra.utils.instantiate(flow_cfg.model, device=dev)
    # state = T.load(f"{path}/checkpoints/last.ckpt", map_location=dev)
    flow = hydra.utils.get_class(flow_cfg.model._target_)
    flow.device=dev
    flow = flow.load_from_checkpoint(f"{path}/checkpoints/last.ckpt", map_location=dev, device=dev)
    return flow

class TransFlow(Flow):
    def __init__(self, data_dims:dict, flow_conf:dict, embed_conf:dict,
                 dense_conf: dict, train_conf:dict, device="cuda", **kwargs):
        super().__init__(train_conf)

        # save hyperparameters
        # self.save_hyperparameters()

        self.data_dims = data_dims
        self.flow_conf = flow_conf
        self.train_conf = train_conf
        self.embed_conf = embed_conf
        self.dense_conf = dense_conf
        self.target_norm = kwargs.get("target_norm", "minmax")
        self.get_network()
        self.to(device)
        
        # validation step outputs
        self.validation_step_outputs = []
        self.validation_step_truth = []
        self.validation_step_posterior = []
        
        # save hp
        self.save_hyperparameters()
        
        # plotting eval framework
        self.eval_f = EvaluateFramework()
        
        
    def get_network(self):
        # Initialise the individual normalisation layers
        self.cnsts_normaliser = IterativeNormLayer((1,self.data_dims["cnts"]), max_iters=50_000)
        self.scalars_normaliser = IterativeNormLayer((1, self.data_dims["scalars"]), max_iters=50_000)
        if self.target_norm=="minmax":
            self.minmax = MinMaxLayer(np.array([[0]]), np.array([[200]]),
                                    feature_range=[-4,4])
        elif self.target_norm=="standard":
            self.minmax = IterativeNormLayer((1, 1), max_iters=50_000)
        else:
            raise ValueError(f"target_norm {self.target_norm} not recognised")

        self.flow = stacked_norm_flow(**self.flow_conf).to(self.device)
        self.embed = self.embed_conf()

    def get_embed(self, ctxt:dict):

        # normalise ctxt
        if "cnts" in ctxt:
            ctxt["cnts"] = self.cnsts_normaliser(ctxt["cnts"].to(self.device),
                                                 mask=ctxt["mask"].to(self.device))

        if "scalars" in ctxt:
            ctxt["scalars"] = self.scalars_normaliser(ctxt["scalars"].to(self.device))

        # run embedding and return
        return self.embed(x=ctxt["cnts"], mask=ctxt["mask"], ctxt=ctxt["scalars"])
    
    def get_flow_ctxt(self, ctxt:dict):
        return self.get_embed(ctxt)
    
    def _shared_step(self, batch:dict, batch_idx:int):

        # narrow gauss to make discrete values continuous
        # smear by N(0,0.5)
        randn = T.rand_like(batch["images"]).to(self.device)*0.5

        # minmax with narrow noise
        x = self.minmax(batch["images"].to(self.device)+randn)

        # run embedding
        if "ctxt" in batch:
            y = self.get_flow_ctxt(batch["ctxt"].copy())

        # get loss
        loss = -self.flow.log_prob(inputs=x, context=y).mean()

        return loss
    
    def sample(self, ctxt:dict=None, n:int=1):

        # get context
        if ctxt is not None:
            ctxt = self.get_flow_ctxt(ctxt)

        # get sample
        samples = self.flow.sample(n, context=ctxt)

        if n==1: # remove sampling dim
            samples = samples.view(-1,1)
        
        # inverse transform
        N = self.minmax.reverse(samples)

        return T.round(N)

    def validation_step(self, batch:dict, batch_idx:int):
        "run validation batches and log results"
        # get val loss
        loss= self._shared_step(batch, batch_idx)
        self.log("valid/loss", loss, prog_bar=True)

        # get val loss
        pred_N = self.sample(batch["ctxt"].copy(), n=1)
        
        # save for end validation
        self.validation_step_outputs.extend(pred_N.cpu().detach())
        self.validation_step_truth.extend(batch["images"].cpu().detach())
        
        # get posterior
        pred_N = self.sample(batch["ctxt"].copy(), n=512)
        self.validation_step_posterior.extend(pred_N.squeeze().cpu().detach())

        return loss

    def on_validation_epoch_end(self):
        """log validation results over all valid batches"""
        all_preds = T.stack(self.validation_step_outputs).numpy()
        all_truth = T.stack(self.validation_step_truth).numpy()
        all_posterior = T.stack(self.validation_step_posterior).numpy()

        error = all_preds-all_truth
        
        # log scalars
        self.log("valid/mae", np.abs(error).mean(), prog_bar=True)

        self.log("valid/RE", (np.abs(error)/all_truth).mean(), prog_bar=True)

        # log distribution
        log={}
        log = self.eval_f.plot_marginals(all_preds, all_truth, col_name=["N"],
                                         log={}, hist_kwargs={
                                             "dist_styles":[{"label":"Predict", "color":"b"},
                                                            {"label":"Truth", "color":"r"}],
                                             "percentile_lst": [1, 99]
                                         })

        log = self.eval_f.plot_marginals(error/all_truth, col_name=["RE"],
                                         log=log, ratio_bool=False,
                                         hist_kwargs={
                                             "dist_styles":[{"label":"(pred-true)/true", "color":"b"}],
                                             "percentile_lst": [1, 99]
                                         })
        
        # truth quantile
        truth_quantile = get_percentile(all_posterior, all_truth,
                                        ["N"], numpy_version=True)
        log = self.eval_f.plot_marginals(truth_quantile,
                                         np.random.uniform(0,100, (100_000,1)),
                                         col_name=["Truth quantile"],
                                         log=log,hist_kwargs={
                                             "dist_styles":[{"label":"Truth quantile", "color":"b"},
                                                            {"label":"Uniform", "color":"r"}],
                                             "percentile_lst": [0, 100],
                                         })
        for i,j in log.items():
            self.logger.log_image(key=i, images=[j])

        self.validation_step_outputs.clear()  # free memory
        self.validation_step_truth.clear()  # free memory
        self.validation_step_posterior.clear()  # free memory
        # if self.current_epoch==0:
        #     self.trainer.fit_loop.epoch_loop.max_steps= self.global_step


@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="flow_config")
def main(config:dict):
    T.set_float32_matmul_precision('medium')

    #dataloader
    train_loader = hydra.utils.instantiate(config.data.train,
                                           loader_config=config.data.loader_config)
    if "jet" in config.data.valid._target_.lower():
        valid_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            max_cnstits=train_loader.dataset.max_cnstits,
                                            datatype=train_loader.dataset.datatype,
                                            )
    else:
        valid_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            )
    # init network
    config.model.train_conf.sch_config.T_max=len(train_loader)*config.model.train_conf.epochs
    network=hydra.utils.instantiate(config.model)
    
    # init callbacks
    callbacks = instantiate_collection(config.callbacks)
    
    # init logger
    wandb = instantiate_collection(config.wandb)

    # train model
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks,
                                      logger=wandb, val_check_interval=len(train_loader)//2)

    trainer.fit(model=network, train_dataloaders=train_loader.train_dataloader(),
                val_dataloaders=valid_loader.test_dataloader())


if __name__ == "__main__":
    main()