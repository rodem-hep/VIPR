import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import sys
import os
import copy

import torch as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import omegaconf
import wandb
import hydra

# internal
from tools import misc
import src.diffusion_schemes as ds
# from src.models.image_model import UNet # used on hydra
from src.utils import fig2img
import src.pipeline as pl

class DiffusionModel(
    # ds.UniformDiffusion
    ds.ElucidatingDiffusion
    ):
    def __init__(self, diffusion_config, network, save_path,
                 train_loader, test_loader, wandb=None, device="cuda",
                 **kwargs):
        super().__init__()
        self.eval_fw = kwargs.get("eval_fw", "")
        self.eval_iters = diffusion_config.get("eval_iters", 1)
        self.wandb=wandb
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device=device
        self.diffusion_config=diffusion_config
        self.n_diffusion_steps = diffusion_config.n_diffusion_steps
        self.save_path=save_path
        
        # mixed precision
        self.loss_scaler = None
        self.loss_scaler = T.cuda.amp.GradScaler()

        # get norms
        self.eval_ctxt =None
        try:
            self.cnts_mean = {i: T.tensor(j).float() for i,j in self.train_loader.dataset.mean.items()}
            self.cnts_std = {i: T.tensor(j).float() for i,j in self.train_loader.dataset.std.items()}
            if self.test_loader is not None:
                self.eval_ctxt = self.test_loader.dataset.get_normed_ctxt()
        except AttributeError:
            self.cnts_mean = self.train_loader.mean
            self.cnts_std = self.train_loader.std
            if self.test_loader is not None:
                self.eval_ctxt = self.test_loader.get_normed_ctxt()
        # self.ctxt_mean = T.tensor(self.train_loader.dataset.ctxt_mean, device=self.device)
        # self.ctxt_std = T.tensor(self.train_loader.dataset.ctxt_std, device=self.device)
        
        self.network = network.to(device)
        self.ema_network = copy.deepcopy(self.network)

        self.optimizer = T.optim.AdamW(self.network.parameters(),
                                       lr=diffusion_config.learning_rate)


        self.log_columns=["epoch", "denoise_images"]
        self.log_columns += [f"{j}_{i}" for i in ["train", "valid"]
                                for j in ["noise_loss", "image_loss"]]
        self.log={i:[] for i in self.log_columns}

        self.loss = nn.MSELoss()


        # eval with same noise
        n_cnts = self.eval_ctxt.pop("true_n_cnts",self.train_loader.dataset.max_cnstits)
        self.initial_noise = pl.generate_gaussian_noise(eval_ctxt=self.eval_ctxt,
                                                        n_constituents=n_cnts,
                                                        **diffusion_config.init_noise)
        
        # save best
        self.noise_loss_best=999

        # save configs
        os.makedirs(save_path, exist_ok=True)
        for i in ["figures", "states"]:
            os.makedirs(f"{save_path}/{i}", exist_ok=True)

        misc.save_yaml(diffusion_config, f"{save_path}/diffusion_cfg.yaml")
        # misc.save_yaml(unet_config, f"{save_path}/unet_cfg.yaml")
        
    def save(self, path, additional_info={}):
        states_to_save = {
            'model': self.ema_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        states_to_save.update(additional_info)
        T.save(states_to_save, path)

    def denormalize(self, images, clip_bool:bool=True):
        # convert the pixel values back to 0-1 range
        images = self.cnts_mean["images"] + images * self.cnts_std["images"]
        return T.clip(images, min, max) if clip_bool else images

    def generate(self, images,  ctxt=None, mask=None):
        # noise -> images -> denormalized images
        with T.no_grad():
            generated_images = self.reverse_diffusion(images=images, ctxt=ctxt, mask=mask)

        generated_images = self.denormalize(generated_images, clip_bool= mask is None)

        return {"gen_data": generated_images, "mask": mask, "ctxt": ctxt}
        # if mask is None:
        #     return generated_images.permute(0, 2, 3, 1)
        # else:
        #     return generated_images, mask
    
    def wandb_log(self):
        self.wandb.log(self.log)
        self.log={i:[] for i in self.log_columns}
    
    def generate_samples(self):

        generated_data={"gen_data":T.tensor([]), "mask":T.tensor([]), "ctxt":{}}

        for sample in tqdm(self.initial_noise, total=len(self.initial_noise),
                            disable=True # len(self.initial_noise)==1
                            ):

            # sample = {i:j.to(self.device) for i,j in sample.items()}

            _generated = self.generate(**sample)

            # concat to generated_data
            for i,j in _generated.items():
                if isinstance(j, dict): # for dict nested ctxt
                    for k,l in j.items():
                        if k not in generated_data[i]:
                            generated_data[i][k]=T.tensor([])
                        
                        generated_data[i][k] = T.concat([generated_data[i][k], l],0)
                else:
                    if T.isnan(_generated[i]).any():
                        print("why")
                    generated_data[i] = T.concat([generated_data[i], _generated[i]],0)

        return generated_data

    def run_evaluate(self, test_loader, epoch_nr=0):

        # plot random generated images for visual evaluation of generation quality
        if (not epoch_nr%self.eval_iters) & True:

            generated_data = self.generate_samples()
            # if isinstance(self.eval_fw, type): # check if eval_fw is class
            log = self.eval_fw(**generated_data)
            self.log.update(log)
            # else:
            #     generated_data, generated_mask = self.plot_images(test_images=(generated_data,generated_mask),
            #                                     epoch=epoch_nr, name="denoise_images")

        if (test_loader is not None) & True:
            noise_loss = {i.replace("_valid", ""):[] for i in self.log.keys()
                          if "valid" in i}
            # run over training samples
            for sample in tqdm(test_loader):

                # sample = {i: j.to(self.device) for i,j in sample.items()}
                with T.no_grad():
                    log_ep = self._shared_step(**sample, training=False)
                for i,j in log_ep.items():
                    noise_loss[i].append(j.cpu().detach().numpy())

            for i,j in noise_loss.items():
                if len(j)>0:
                    self.log[f"{i}_valid"] = np.mean(j)
                
    def run_training(self, run_eval=True):
        
        #progress bar for training
        pbar = tqdm(range(self.diffusion_config.num_epochs))


        for ep in pbar:

            if run_eval:
                # run evaluation
                self.run_evaluate(self.test_loader, ep)

            # run over training samples
            for nr, sample in enumerate(self.train_loader):

                # sample = {i: j.to(self.device) for i,j in sample.items()}

                log_ep = self.train_step(**sample)
                
                for key, items in log_ep.items():
                    if np.isnan(items):
                        raise ValueError("is NaN")
                    self.log[key+"_train"].append(items)
                    
            # input logging
            for key in log_ep:
                self.log[key+"_train"] = np.mean(self.log[key+"_train"])
            
            if run_eval:
                if self.log["noise_loss_valid"]<self.noise_loss_best:
                    self.save(f"{self.save_path}/states/diffusion_{ep}.pth")
                    self.noise_loss_best = self.log["noise_loss_valid"]

            self.log["epoch"] = ep

            if self.wandb is not None:
                self.wandb_log()
            

    def plot_images(self, test_images, epoch=None, num_rows=3, num_cols=3, name=""):

        if len(test_images)==2:
            test_images, mask = test_images

        if len(test_images.shape) == 4:
            test_images= test_images.permute(0, 2, 3, 1)

        if isinstance(test_images, T.Tensor):
            mask = mask.cpu().detach().numpy()
            test_images = test_images.cpu().detach().numpy()

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(8*num_rows, 6*num_cols))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                    
                if "pc" in self.diffusion_config.init_noise.datatype:
                    ax[row, col].scatter(test_images[index, mask[index], 0],
                                test_images[index, mask[index], 1],
                                s=test_images[index, mask[index], 2],
                                marker="o"
                                )
                else:
                    style = {"vmin":0, "vmax":1}
                    if test_images[index].shape[-1]:
                        style["cmap"] = "gray"
                    ax[row, col].imshow(test_images[index], **style)
                    ax[row, col].axis("off")
        plt.tight_layout()
        plt.title(epoch)
        if "pc" in self.diffusion_config.init_noise.datatype:
            fig = fig2img(fig)
            fig = wandb.Image(fig)
        self.log[name] = fig
        plt.close()
        return test_images, mask

def get_standardization(loader, pc:bool=False):
    "dummy standardization function"
    data=[]
    data_ctxt=[]
    mask_lst = []
    for nr, i in tqdm(enumerate(loader)):
        if pc:
            data.append(i["images"])
            mask_lst.append(i["mask"])
            data_ctxt.append(i["images"])
        else:
            data.append(i[0] if len(i) else i)
            if len(i[1].shape)>2:
                data_ctxt.append(i[1] if len(i) else i)
            else:
                data_ctxt.append(None)
        if nr==10:
            break
    data=T.concat(data)
    if data_ctxt[0] is not None:
        data_ctxt=T.concat(data_ctxt)
    if len(mask_lst)>0:
        mask_lst=T.concat(mask_lst)
        
        return (data[:9], mask_lst[:9]), data_ctxt, None,None
    else:
        return data, data_ctxt, data.mean(), data.std()

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config")
def main(config):
    #dataloader
    train_loader = hydra.utils.instantiate(config.data_cfgs.train,
                                           loader_config=config.data_cfgs.loader_config)
    test_loader = hydra.utils.instantiate(config.data_cfgs.valid,
                                          loader_config=config.data_cfgs.loader_config,
                                          max_cnstits=train_loader.max_cnstits)

    #dataloader
    if "Jet" in test_loader.__str__():
        eval_fw = test_loader
    else:
        eval_fw = hydra.utils.instantiate(config.eval)

    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    wandb.init(config=config, **config.wandb)

    # init network
    network=hydra.utils.instantiate(config.model,ctxt_dims=train_loader.get_ctxt_shape())
    diffusion_config = hydra.utils.instantiate(config.diffusion_cfg)
    diffusion_config.init_noise.shape = list(train_loader._shape())

    # init diffusion
    model = DiffusionModel(diffusion_config=diffusion_config,
                           network=network,
                           train_loader=train_loader.train_dataloader(),
                           test_loader=test_loader.test_dataloader(),
                           device=config.device,
                           save_path=config.save_path,
                           eval_fw=eval_fw,
                           wandb=wandb, 
                           )

    print(f"Trainable parameters: {model.network.count_trainable_parameters()}")
    wandb.config.update({"Model Parameters": model.network.count_trainable_parameters()})
    
    if config.diffusion_cfg.super_res & (model.wandb is not None):
        #find std and mean of data
        data, data_ctxt, _, _ = get_standardization(train_loader,
                                                pc = "PointCloud" in train_loader.__str__())
        model.plot_images(test_images=data_ctxt[:9], name="context_image")
        model.plot_images(test_images=data[:9], name = "true_image")
    # else:
    #     model.plot_images(test_images=data, name="true_image")
    #     # model.plot_images(name="context_image")
    #     data_ctxt=[None]

    model.run_training(run_eval=True)

        
if __name__ == "__main__":
    main()
