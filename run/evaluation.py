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
import pandas as pd


from tools import misc
import src.physics as phy 
from tools.datamodule.pipeline import pc_2_image
from tools.datamodule.prepare_data import fill_data_in_pc, matrix_to_point_cloud
import src.eval_utils as eutils
import src.diffusion_schemes as ds

hist_kwargs={"style":{"bins": 50},
             "dist_styles":[{"label": r"jet$_{truth}$"},
                            {"label": r"jet$_{diffusion}$"}],
                "percentile_lst":[1,99],
                }

class EvaluatePhysics(eutils.EvaluateFramework):
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
        self.eval_files = glob(f"{self.eval_folder}/*.csv")
        self.eval_files_available = len(self.eval_files)>0
        
        # change values in config
        if "eval_cfg" in kwargs:
            self.config.trainer.eval_cfg.update(kwargs["eval_cfg"])
        self.config.trainer.init_noise.size = size
        self.loader_config = kwargs.get("loader_config", self.config.data.loader_config)

    def get_eval_files(self, specific_file=None):
        eval_data = {}
        eval_files = self.eval_files
        if specific_file is not None:
            eval_files = [i for i in eval_files if specific_file in i]
            
        for i in eval_files:
            name = i.split(".csv")[0].split("/")[-1]
            eval_data[name] = pd.read_csv(i)
        return eval_data

    def load_diffusion(self):
        #get network    
        network = hydra.utils.instantiate(self.config.model,
                                          ctxt_dims=self.diffusion_cfg["ctxt_dims"])
        network.eval()
        
        #set dataloader to testing
        data_cfg = self.config.data.valid
        data_cfg["sub_col"] = "test"
        data_cfg["n_files"]=10
        data_cfg["loader_config"] = self.loader_config
        self.data = hydra.utils.instantiate(data_cfg)

        # load old state
        self.checkpoint_file = misc.sort_by_creation_time(
            glob(f"{self.path_to_model}/states/dif*"))[-1]

        # load weights
        self.checkpoint = T.load(self.checkpoint_file, map_location=self.device)
        
        #normaliser are saved in incorrect shape
        self.checkpoint.update({i:j.flatten() for i,j in self.checkpoint.items()
                                if "normaliser" in i})
        
        # load diffusion framework
        self.diffusion = hydra.utils.instantiate(
            self.config.trainer,
            network=network,
            inpt_shape=self.data._shape(),
            device=self.device,
            loader_config=self.loader_config,
            )
        
        # load state
        try:
            self.diffusion.load_state_dict(self.checkpoint)
        except:
            pass
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
        gen_cnts = self.data.relative_pos(**gen_data_rel, reverse=True)
        
        # get gen jet properties
        gen_jet_vars = self.data.physics_properties(
            gen_cnts, gen_data_rel["mask"])

        return gen_jet_vars, gen_data_rel, gen_cnts, gen_data_rel["mask"]
    
    def generate_post(self, ctxt:dict, repeats:int=10):
        
        # for k in range(len(ctxt["cnts"])):
        eval_ctxt = {}
        for i,j in ctxt.items():
            eval_ctxt[i] = np.repeat(j, repeats=repeats, axis=0)

        n_cnts = eval_ctxt.pop("true_n_cnts")

        self.diffusion.init_noise["size"]=len(n_cnts)
        loader_config = self.loader_config.copy()
        loader_config.update({"batch_size": len(n_cnts)})
                             
        initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
                                                n_constituents=n_cnts,
                                                loader_kwargs=loader_config,
                                                **self.diffusion.init_noise)
        return self.generate_sample(initial_noise, disable_bar=True)
        
    def plot_posterier(self, truth, posterier, columns, **kwargs):
        for i in columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # plot ratio between distribution
            counts_dict, _ = plot.plot_hist(posterier[i].values, ax=ax, **kwargs)
            ax.axvline(truth[i], label="Truth", color="black")
            ax.set_xlabel(i)
            
    def generate_and_save_post(self, eval_ctxt, n_post_to_gen, saving_name="",
                               combine_ctxt_size=1):
        if n_post_to_gen>len(eval_ctxt["scalars"]):
            raise ValueError("n_post_to_gen too large")
        
        # create lst for logging
        # dist_percentiles = {i: [] for i in jet_vars}
        all_gen_cnts=[]
        all_gen_jets=[]
        
        # run n number of ctxt

        if n_post_to_gen>1:
            saving_name = f"posterior_{self.size}"
            jet_size= self.size
            self.data.jet_vars.iloc[:jet_size].to_csv(f"{self.eval_folder}/truth_jets_{saving_name}.csv")
            steps=np.arange(self.size+combine_ctxt_size, step=combine_ctxt_size)
            for nr, (lower, high) in tqdm(enumerate(zip(steps[:-1], steps[1:])),
                                          total=len(steps)):
                # get ctxt
                ctxt = {i:j[lower:high] for i,j in eval_ctxt.items()}
                # ctxt["cnts"] = np.expand_dims(ctxt["cnts"][ctxt["mask"]],0)
                
                # generate sample
                gen_jet_vars, _, gen_cnts, gen_cnts_mask  = self.generate_post(ctxt, n_post_to_gen)

                # posterior nr
                n_post = (gen_cnts_mask*np.r_[np.ones((n_post_to_gen,1))*lower,np.ones((n_post_to_gen,1))*(high-1)])
                
                # correct index per event
                new_index = (high-lower)*n_post_to_gen*nr
                index = gen_cnts_mask*np.arange(len(gen_cnts_mask))[:, None]+new_index
                all_gen_cnts.append(np.c_[gen_cnts[gen_cnts_mask].numpy(),
                                          index[gen_cnts_mask],
                                          n_post[gen_cnts_mask]])

                # eventnumber for identification
                gen_jet_vars["eventNumber"] = np.ravel(np.arange(lower, high)[:,None]*np.ones((1,n_post_to_gen)))
                gen_jet_vars.index = gen_jet_vars.index+new_index

                # log the values
                all_gen_jets.append(gen_jet_vars)
                
                #save samples
                pd.concat(all_gen_jets).to_csv(f"{self.eval_folder}/gen_jets_{saving_name}.csv")
                pd.DataFrame(np.concatenate(all_gen_cnts, 0),
                             columns=["eta", "phi", "pt", "eventNumber", "n_post"]).to_csv(f"{self.eval_folder}/gen_cnts_{saving_name}.csv")
        else:
            n_cnts = eval_ctxt.pop("true_n_cnts")
            initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
                                                        n_constituents=n_cnts,
                                                        loader_kwargs=self.loader_config,
                                                        **self.diffusion.init_noise)
            all_gen_jets, gen_data_rel, all_gen_cnts, mask = self.generate_sample(initial_noise)
            index = mask*np.arange(len(mask))[:, None]
            all_gen_cnts = pd.DataFrame( np.c_[all_gen_cnts[mask].numpy(),index[mask]],
                                        columns=["eta", "phi", "pt", "eventNumber"])
            saving_name="single"
            jet_size= len(all_gen_jets)
            # save the generated data
            if self.save_generated:
                all_gen_jets.to_csv(f"{self.eval_folder}/gen_jets_{saving_name}.csv")
                self.data.jet_vars.iloc[:jet_size].to_csv(f"{self.eval_folder}/truth_jets_{saving_name}.csv")
                all_gen_cnts.to_csv(f"{self.eval_folder}/gen_cnts_{saving_name}.csv")
                
            # TODO truth_cnts should also be added self.data.cnts_vars[:n_points]

        return all_gen_jets, all_gen_cnts


if __name__ == "__main__":
    config = misc.load_yaml(str(root/"configs/evaluate.yaml"))
    eval_fw = hydra.utils.instantiate(config.eval)
    eval_fw.load_diffusion()
    save_figs = config.save_figures
    save_path = f"{config.eval.path_to_model}/figures/"
    
    ### generate sample ###
    eval_ctxt = eval_fw.data.get_normed_ctxt()
    if config.generate_sample:
        gen_cnts, gen_cnts = eval_fw.generate_and_save_post(eval_ctxt,
                                                            # config.jet_vars,
                                                            config.n_post_to_gen,
                                                            combine_ctxt_size=config.combine_ctxt_size)
        sys.exit()
    elif eval_fw.eval_files_available: # load save generated data
        gen_data = eval_fw.get_eval_files()
        truth_jets = gen_data[f"truth_jets_{config.csv_sample_to_load}"]
        mask_mass = truth_jets["mass"]!=0 # some masses are zero?
        obs_jets = pd.DataFrame(eval_ctxt["scalars"][:len(truth_jets)][mask_mass],
                                columns=eval_fw.config.data.train.jet_physics_cfg.jet_scalars_cols).reset_index(drop=True)
        truth_jets=truth_jets[mask_mass].reset_index(drop=True)
        gen_cnts = gen_data[f"gen_cnts_{config.csv_sample_to_load}"].reset_index(drop=True)
        gen_jets = gen_data[f"gen_jets_{config.csv_sample_to_load}"][mask_mass].reset_index(drop=True)
    else:
        raise ValueError("Eval files not avaliable - should be generated and saved!")

    # get truth jets
    truth_jets = eval_fw.data.jet_vars[:len(gen_jets)]
    truth_cnts = eval_fw.data.cnts_vars[:len(gen_jets)][
        eval_fw.data.mask_cnts[:len(gen_jets)]
        ]

    # create pc
    gen_cnts, mask = matrix_to_point_cloud(gen_cnts[["eta", "phi", "pt"]].values,
                                            gen_cnts["eventNumber"].values,
                                        #   num_per_event_max=max_cnts
                                            )

    hist_kwargs={"style":{"bins": 50},"dist_styles":[{"label": "Truth"},
                                                     {"label": "Generated"}],
                 "percentile_lst":[0.1,99.9],
                #  "log_yscale":True
                 }

    ### plot figures ##
    if "single" in config.csv_sample_to_load: # single generated value
        # plot 1d marginals of cnts
        hist_kwargs["dist_styles"] = [{"label": r"jet$_{truth}$"}, 
                                       {"label": r"jet$_{diffusion}$"},
                                    #    {"label": r"jet$_{obs.}$"},
                                       ]
        col_to_plot = ["eta", "phi", "pt", "mass"]
        eval_fw.plot_marginals(truth_jets[col_to_plot].values,
                                gen_jets[col_to_plot].values,
                                # obs_jets[col_to_plot].iloc[:len(truth_jets)].values,
                                col_name=col_to_plot,
                                hist_kwargs=hist_kwargs,
                                save_path=f"{save_path}/gen_jets_" if save_figs else None)

        hist_kwargs["dist_styles"] = [
            {"label": r"(jet$_{diffusion}$-jet$_{truth}$)/jet$_{truth}$"},
            {"label": r"(jet$_{observed}$-jet$_{truth}$)/jet$_{truth}$"}
            ]
        diff_obs = ((obs_jets-truth_jets)/truth_jets)[col_to_plot].iloc[:len(truth_jets)].values
        
        eval_fw.plot_marginals(((gen_jets-truth_jets)/truth_jets)[col_to_plot].values,
                               diff_obs,#-diff_obs.mean(0),
                                col_name=col_to_plot,
                                hist_kwargs=hist_kwargs,
                                save_path=f"{save_path}/diff_jets_" if save_figs else None)
        
        # plot 1d marginals of cnts
        mask_truth = truth_cnts[:,-1]< 1000000
        mask_gen = gen_cnts[mask][:,-1]< 1000000
        hist_kwargs["dist_styles"] = [{"label": r"(cnts$_{truth}$-cnts$_{diffusion}$)/cnts$_{truth}$"
                                       }]
        abs_error_jet = (truth_cnts[mask_truth]-gen_cnts[mask][mask_gen])
        abs_error_jet[:, 1] = phy.rescale_phi(abs_error_jet[:, 1])
        eval_fw.plot_marginals(abs_error_jet/truth_cnts[mask_truth],
                                col_name=["eta", "phi", "pt"],
                                hist_kwargs=hist_kwargs,
                                save_path=f"{save_path}/diff_cnts_" if save_figs else None)

        #### plot #-leading cnts
        truth_index_sort = np.argsort(eval_fw.data.cnts_vars[...,-1],-1)[...,::-1]
        # truth_cnts_sorted = eval_fw.data.cnts_vars[np.argsort(eval_fw.data.cnts_vars[...,-1],-1)]
        gen_index_sort = np.argsort(gen_cnts[...,-1],-1)[...,::-1]
        if False: # # leading cnts

            # hist_kwargs["dist_styles"] = [{"label": r"cnts$_{truth}$"},
            #                               {"label": r"cnts$_{diffusion}$"
            #                             }]
            for i in range(100):
                first_truth_cnts = eval_fw.data.cnts_vars[np.arange(0, len(truth_index_sort),1),
                                                        truth_index_sort[:, i], :]
                first_gen_cnts = gen_cnts[np.arange(0, len(gen_index_sort),1), gen_index_sort[:, i], :]
                hist_kwargs["legend_kwargs"]={"title": f"{i+1} leading cnts"}
                hist_kwargs["percentile_lst"] = [1,99]
                eval_fw.plot_marginals((first_truth_cnts[:len(first_gen_cnts),-1:]-first_gen_cnts[:,-1:])/first_truth_cnts[:len(first_gen_cnts),-1:],
                                        col_name=["pt"],
                                        # col_name=["eta", "phi", "pt"],
                                        hist_kwargs=hist_kwargs,
                                        save_path=f"{save_path}/{i+1}_leading_cnts_" if save_figs else None)
    elif "posterior" in config.csv_sample_to_load: # posterior 
        # plot posterior
        percentile_dict = eutils.get_percentile(gen_jets, truth_jets,
                                                columns=config.jet_vars)
            
        fig, ax = plt.subplots(1,len(config.jet_vars),
                               figsize=(len(config.jet_vars)*8,6))
        for nr, i in enumerate(percentile_dict.keys()):
            ax[nr].hist(percentile_dict[i], range=[0,100], bins=30,
                     label="Posterior")
            ax[nr].set_xlabel(i)
        plt.legend()
        
        # plot posteriors
        for i in range(3):
            mask_evt = gen_jets["eventNumber"]==i
            for col in ["mass", "pt"]: # ["eta", "phi"]:#
                fig, ax = plt.subplots(1,2, figsize=(1.5*8,6))
                ax[0].hist(gen_jets[mask_evt][col], bins=30)
                ax[0].axvline(truth_jets[col].iloc[i], color="black", ls="dashed")
                
                var_srt = np.sort(gen_jets[mask_evt][col])
                ax[1].plot(var_srt, var_srt.cumsum()/np.cumsum(var_srt)[-1])
                ax[1].axvline(truth_jets[col].iloc[i], color="black", ls="dashed")
                ax[1].axhline(percentile_dict[col][i]/100, color="black", ls="dashed")
                plt.tight_layout()
                for k in range(len(ax)):
                    ax[k].set_xlabel(col)
        
        post_width, x_value_of_width = eutils.get_spread_of_post(gen_jets, truth_jets,
                                                               variables=["mass", "pt"],
                                                               norm_width=True)
        eutils.plot_post_spread(post_width, x_value_of_width, var_names = ["mass", "pt"],
                               bins_wth=10)
        
    if config.plot_images: # generate a single value per ctxt
        # n_cnts = eval_ctxt.pop("true_n_cnts")
        # initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
        #                                             n_constituents=n_cnts,
        #                                             loader_kwargs=config.eval.loader_config,
        #                                             **eval_fw.diffusion.init_noise)
        # # generate relative cnts
        # gen_jet_vars, gen_data_rel, gen_cnts, gen_mask = eval_fw.generate_sample(initial_noise)
        # eval_fw.plot_marginals(eval_fw.data.jet_vars.values, gen_jet_vars.values, col_name=gen_jet_vars.columns)
        
        style={"range":[[-2.5, 2.5], [-np.pi, np.pi]], "bins":64}
        ctxt_cnts = phy.relative_pos(eval_ctxt["cnts"][:100],
                                    eval_ctxt["scalars"][:100],
                                    mask=eval_ctxt["mask"][:100], reverse=True)
        
        gen_images = np.clip(np.log(pc_2_image(gen_cnts[:100], style)+1), 0, 1)
        ctxt_images = np.clip(np.log(pc_2_image(ctxt_cnts, style)+1), 0, 1)
        truth_images = np.clip(np.log(pc_2_image( eval_fw.data.cnts_vars[:100], style)+1), 0, 1)
        # ctxt_images = np.clip(np.log(pc_2_image(eval_ctxt["cnts"][:100], style)+1), 0, 1)
        
        
        eval_fw.plot_images(gen_images[..., None], name="gen_image", wandb_bool=False,
                            save_path=f"{save_path}/gen_images_" if save_figs else None)
        eval_fw.plot_images(ctxt_images[..., None], name="gen_image", wandb_bool=False,
                            save_path=f"{save_path}/ctxt_images_"if save_figs else None)
        eval_fw.plot_images(truth_images[..., None], name="gen_image", wandb_bool=False,
                            save_path=f"{save_path}/truth_images_"if save_figs else None)

