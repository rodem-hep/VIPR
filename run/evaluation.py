"evaluate diffusion performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from glob import glob
import hydra
from tqdm import tqdm
import torch as T
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tools.visualization import general_plotting as plot
import scipy.stats as stats
import pandas as pd


from tools import misc
from tools.datamodule.pipeline import pc_2_image 
from src.eval_utils import EvaluateFramework
import src.diffusion_schemes as ds

class EvaluatePhysics(EvaluateFramework):
    def __init__(self, path_to_model, size=None, device="cuda", **kwargs):
        self.path_to_model=path_to_model
        self.device=device
        self.size=size
        self.save_generated=kwargs.get("save_generated", False)
        self.config = misc.load_yaml(path_to_model+"/.hydra/config.yaml")
        self.diffusion_cfg = misc.load_yaml(path_to_model+"/diffusion_cfg.yaml")
        
        # create eval folder
        self.eval_folder = f"{path_to_model}/eval_files/"
        os.makedirs(self.eval_folder, exist_ok=True)
        self.eval_files = glob(f"{self.eval_folder}/*")
        self.eval_files_available = len(self.eval_files)>0
        
        # change values in config
        if "eval_cfg" in kwargs:
            self.config.trainer.eval_cfg.update(kwargs["eval_cfg"])
        self.config.trainer.init_noise.size = size
        self.loader_config = kwargs.get("loader_config", self.config.data.loader_config)
        
        self.load_diffusion()

    def get_eval_files(self):
        eval_data = {}
        for i in self.eval_files:
            name = i.split(".csv")[0].split("/")[-1]
            eval_data[name] = pd.read_csv(i)
        return eval_data

    def load_diffusion(self):
        #get network    
        network = hydra.utils.instantiate(self.config.model,
                                          ctxt_dims=self.diffusion_cfg["ctxt_dims"])
        network.eval()
        
        #get dataloader
        data_cfg = self.config.data.valid
        data_cfg["sub_col"] = "test"
        self.config.data.train.jet_physics_cfg["loader_config"] = self.loader_config
        self.test_loader = hydra.utils.instantiate(self.config.data.train,
                                            loader_config=self.loader_config
                                            )
        self.data = self.test_loader.dataset
        
        # load old state
        self.checkpoint_file = misc.sort_by_creation_time(
            glob(f"{self.path_to_model}/states/dif*"))[-1]

        # load weights
        self.checkpoint = T.load(self.checkpoint_file)
        
        #normaliser are saved in incorrect shape
        self.checkpoint.update({i:j.flatten() for i,j in self.checkpoint.items()
                                if "normaliser" in i})
        
        # load diffusion framework
        self.diffusion = hydra.utils.instantiate(
            self.config.trainer,
            network=network,
            inpt_shape=self.test_loader.dataset._shape(),
            device=self.device,
            loader_config=self.loader_config,
            )
        
        # load state
        self.diffusion.load_state_dict(self.checkpoint)

        # set to eval
        self.diffusion.eval()

    def generate_sample(self, initial_noise, disable_bar=False):
        # generate relative cnts
        # print("Generating samples:")
        generated_data_rel = self.diffusion.generate_samples(initial_noise,
                                                             disable_bar=disable_bar)

        # define dict for relative_pos
        gen_data_rel = {}
        gen_data_rel["cnts_vars"] = generated_data_rel["gen_data"]
        gen_data_rel["jet_vars"] = generated_data_rel["ctxt"]["scalars"]
        gen_data_rel["mask"] = generated_data_rel["mask"]==1

        # relative cnts to cnts
        gen_cnts = self.test_loader.dataset.relative_pos(**gen_data_rel, reverse=True)
        
        # get gen jet properties
        gen_jet_vars = self.test_loader.dataset.physics_properties(
            gen_cnts, gen_data_rel["mask"]) 

        return gen_jet_vars, gen_data_rel, gen_cnts, gen_data_rel["mask"]
    
    def generate_post(self, ctxt:dict, repeats:int=10):
        
        for k in range(len(ctxt["cnts"])):
            eval_ctxt = {}
            for i,j in ctxt.items():
                eval_ctxt[i] = np.repeat(j[k:k+1], repeats=repeats,
                                         axis=0)
            n_cnts = eval_ctxt.pop("true_n_cnts")
            initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
                                                    n_constituents=n_cnts,
                                                    loader_kwargs=self.loader_config,
                                                    **self.diffusion.init_noise)
            yield self.generate_sample(initial_noise, disable_bar=True)
        
    def plot_posterier(self, truth, posterier, columns, **kwargs):
        for i in columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # plot ratio between distribution
            counts_dict, _ = plot.plot_hist(posterier[i].values, ax=ax, **kwargs)
            ax.axvline(truth[i], label="Truth", color="black")
            ax.set_xlabel(i)
            
    def generate_and_save_post(self, eval_ctxt, jet_vars, n_points):
        if n_points>len(eval_ctxt["scalars"]):
            raise ValueError("n_points too large")
        
        # create lst for logging
        # dist_percentiles = {i: [] for i in jet_vars}
        all_gen_cnts=[]
        all_gen_jets=[]
        
        # run n number of ctxt
        for nr in tqdm(range(n_points)):
            # get ctxt
            ctxt = {i:j[nr:nr+1] for i,j in eval_ctxt.items()}
            # ctxt["cnts"] = np.expand_dims(ctxt["cnts"][ctxt["mask"]],0)
            
            # generate sample
            posterior  = self.generate_post(ctxt, 512)
            gen_jet_vars, _, gen_cnts, gen_cnts_mask = next(posterior)
            
            # correct index per event
            new_index = nr*512
            index = gen_cnts_mask*np.arange(len(gen_cnts_mask))[:, None]+new_index
            gen_jet_vars["eventNumber"] = nr
            gen_jet_vars.index = gen_jet_vars.index+new_index

            # log the values
            all_gen_cnts.append(np.c_[gen_cnts[gen_cnts_mask].numpy(),
                                      index[gen_cnts_mask]])
            
            all_gen_jets.append(gen_jet_vars)
        #concat
        all_gen_jets = pd.concat(all_gen_jets)
        all_gen_cnts = pd.DataFrame(np.concatenate(all_gen_cnts, 0),
                                    columns=["eta", "phi", "pt", "eventNumber"])
        # save the generated data
        if self.save_generated:
            all_gen_jets.to_csv(f"{self.eval_folder}/gen_jets.csv")
            self.data.jet_vars.iloc[:n_points].to_csv(f"{self.eval_folder}/truth_jets.csv")
            all_gen_cnts.to_csv(f"{self.eval_folder}/gen_cnts.csv")
            
            # TODO truth_cnts should also be added self.data.cnts_vars[:n_points]

        return all_gen_jets, all_gen_cnts
    
    def get_percentile(self, gen_jet_vars, truth_jet_vars, columns):
        percentile_dict = {i:[] for i in columns}
        for i in np.unique(gen_jet_vars.eventNumber):
            for key in percentile_dict:
                mask_evt = gen_jet_vars["eventNumber"]==i
                percentile_dict[key].append(
                    stats.percentileofscore(gen_jet_vars[key][mask_evt],
                                            truth_jet_vars[key].iloc[i])
                    )
        return percentile_dict

if __name__ == "__main__":
    config = misc.load_yaml(str(root/"configs/evaluate.yaml"))
    eval_fw = hydra.utils.instantiate(config.eval)
    
    # generate sample
    eval_ctxt = eval_fw.test_loader.dataset.get_normed_ctxt()
    if False: # plots cnts marginal 
        n_cnts = eval_ctxt.pop("true_n_cnts")
        initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
                                                    n_constituents=n_cnts,
                                                    loader_kwargs=config.eval.loader_config,
                                                    **eval_fw.diffusion.init_noise)
        # generate relative cnts
        gen_jet_vars, gen_cnts_rel, gen_cnts = eval_fw.generate_sample(initial_noise)
        eval_fw.plot_marginals(eval_fw.data.jet_vars.values, gen_jet_vars.values, col_name=gen_jet_vars.columns)
        
        style={"range":[[-2.5, 2.5], [-np.pi, np.pi]], "bins":64}
        
        gen_images = np.clip(np.log(pc_2_image(gen_cnts[:100].numpy(), style)+1), 0, 1)
        ctxt_images = np.clip(np.log(pc_2_image(eval_ctxt["cnts"][:100], style)+1), 0, 1)
        
        
        eval_fw.plot_images(gen_images[..., None], name="gen_image", wandb_bool=False)
        eval_fw.plot_images(ctxt_images[..., None], name="gen_image", wandb_bool=False)
    elif eval_fw.eval_files_available: # load save generated data
        gen_data = eval_fw.get_eval_files()
        truth_jets = gen_data["truth_jets"]
        gen_cnts = gen_data["gen_cnts"]
        gen_jets = gen_data["gen_jets"]
    else: # generate new data
        gen_jets, gen_cnts = eval_fw.generate_and_save_post(eval_ctxt,
                                                                    config.jet_vars,
                                                                    config.n_post_to_gen)
        truth_jets = eval_fw.data.jet_vars
        
    percentile_dict = eval_fw.get_percentile(gen_jets, truth_jets,
                                                columns=config.jet_vars)
        
    for i in percentile_dict.keys():
        plt.figure()
        plt.hist(percentile_dict[i], range=[0,100], bins=10)
        plt.xlabel(i)


    # truth_jet_vars = eval_fw.data.jet_vars.iloc[:4]
    # gen_jet_vars = all_gen_jets
    
    # percentile_dict = {i:[] for i in config.jet_vars}
    # for i in np.unique(gen_jet_vars.eventNumber):
    #     for key in percentile_dict:
    #         mask_evt = gen_jet_vars["eventNumber"]==i
    #         percentile_dict[key].append(
    #             stats.percentileofscore(gen_jet_vars[key][mask_evt],
    #                                     eval_fw.data.jet_vars.iloc[i][key])
    #             )
        
    

    # posterior  = eval_fw.generate_post(
    #     {i:j[:1] for i,j in eval_ctxt.items()}, 1024)
    # gen_jet_vars, gen_cnts_rel, gen_cnts = next(posterior)
    
    
    # diffs = (gen_jet_vars-eval_fw.data.jet_vars.iloc[0]).drop(columns=["n_cnts"])
    # # eval_fw.plot_marginals(diffs.values,
    #                     #    col_name=[fr"$\Delta${i}" for i in diffs.columns][:4],
    #                     #    hist_kwargs={"style":{"bins":25},
    #                     #                 "percentile_lst": [1,99]})
    # eval_fw.plot_posterier(eval_fw.data.jet_vars.iloc[0], gen_jet_vars,
    #                        columns=config.jet_vars,
    #                        dist_styles=[{"label":"Posterier"}])
    # for i in config.jet_vars:
    #     print(stats.percentileofscore(gen_jet_vars[i], eval_fw.data.jet_vars.iloc[0][i]))
    
        
        
        
        