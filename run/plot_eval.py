"evaluate diffusion performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from glob import glob
import hydra
from tqdm import tqdm
from typing import List
import torch as T
import numpy as np
import os
import h5py
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from run.run_flow import load_flow_from_path

# import src.physics as phy 
import src.eval_utils as eutils
import src.diffusion_schemes as ds
from run.eval_flow import DictDataset

from tools import misc, hydra_utils

from tools.visualization import general_plotting as plot
from tools.datamodule.prepare_data import matrix_to_point_cloud
import tools.physics as phy

hist_kwargs={"style":{"bins": 50},
             "dist_styles":[{"label": r"jet$_{Top}}$"},
                            {"label": r"jet$_{diffusion}$"}],
                "percentile_lst":[1,99],
                }

def get_pileup_name(pileup_dict):
    # pileup naming
    name=""
    if pileup_dict != "":
        name = "_pileup_"+"_".join([f"{i}_{j}" for i,j in pileup_dict.items()])
    return name

class EvaluatePhysics(eutils.EvaluateFramework):
    def __init__(self, path_to_model, size=None, device="cuda", **kwargs):
        self.path_to_model=path_to_model
        self.device=device
        self.size=size
        self.verbose=kwargs.get("verbose", False)
        self.format=kwargs.get("format", ".png")
        self.config = misc.load_yaml(path_to_model+"/.hydra/config.yaml")
        self.diffusion_cfg = misc.load_yaml(path_to_model+"/diffusion_cfg.yaml")
        self.flow_path = kwargs.get("flow_path")
        self.run_flow = kwargs.get("run_flow", False)
        
        # 
        self.eval_folder = f"{path_to_model}/eval_files/"
        
        # account for flow model
        if  self.flow_path  is not None:
            # create new eval folder for flow tests
            self.eval_folder += f"/flow_N/{self.flow_path}"
            self.flow_path = "/".join(path_to_model.split("/")[:-2])+f"flow/{self.flow_path}/"
            # load flow model
            self.flow = load_flow_from_path( self.flow_path, device)

        # create eval folder
        os.makedirs(self.eval_folder, exist_ok=True)
        
        self.eval_files = glob(f"{self.eval_folder}/*.csv")+glob(f"{self.eval_folder}/*.h5")
        self.eval_files_available = len(self.eval_files)>0
        
        # change values in config
        if "eval_cfg" in kwargs:
            self.config.trainer.eval_cfg.update(kwargs["eval_cfg"])
        self.config.trainer.init_noise.size = size
        self.loader_config = kwargs.get("loader_config", self.config.data.loader_config)
        
        #set dataloader to testing
        data_cfg = self.config.data.valid
        data_cfg["sub_col"] = "test"
        if self.size is not None:
            data_cfg["n_files"]=kwargs.get("n_files", (self.size//10_000))
            data_cfg["n_files"] += 1 if data_cfg["n_files"]==0 else 0
        else:
            data_cfg["n_files"]=None
        data_cfg["loader_config"] = self.loader_config
        data_cfg.update(kwargs.get("data_cfg", {}))
        self.config.data.train.jet_physics_cfg.pileup_path[0] = self.config.data.train.jet_physics_cfg.pileup_path[0].replace("/home/users/a/algren/scratch",
                                                                                                                              "/srv/beegfs/scratch/users/a/algren/")
        self.data = hydra.utils.instantiate(data_cfg)


    def get_eval_files(self, should_contain:str="", specific_file:str=".csv",
                       eval_files:List[str]=None):
        eval_data = {}
        eval_files = self.eval_files if eval_files is None else eval_files
        if specific_file is not None:
            eval_files = [i for i in eval_files if specific_file in i]

        # eval_files = [i for i in eval_files if "jet_substructure" not in i]
            
        for i in eval_files:
            name = i.split(".")[0].split("/")[-1]

            if should_contain not in name:
                continue

            names=None
            if "cnts" in name:
                names=["eta", "phi", "pt", "eventNumber", "n_post"]
            elif "jets" in name:
                names=["px", "py", "pz", "mass", "pt", "eta", "phi", "n_cnts"]

            if "csv" in i:
                import csv
                with open(i, newline='') as csvfile:
                    spamreader = csv.reader(csvfile,delimiter=",")
                    # print(spamreader[names])
                    data=list(spamreader)
                columns = data[0][1:]
                data = np.array(data[1:])[:,1:]
                if self.verbose:
                    print(name)
                    print(len(data))
                if np.any(data==""):
                    data[np.any(data=="",1)] = np.nan
                eval_data[name]  = pd.DataFrame(data, columns=columns).astype(np.float32)
            else:
                with h5py.File(i, 'r+') as hf:
                    eval_data[name] = hf["gen_cnts"][:]

        return eval_data

    def load_diffusion(self):
        #get network    
        network = hydra.utils.instantiate(self.config.model,
                                          ctxt_dims=self.diffusion_cfg["ctxt_dims"])
        network.eval()
        
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
        if self.run_flow:
            # change n_cnts to the flow predicted one
            sample = {i: T.tensor(j, device=self.flow.device) for i,j in eval_ctxt.items()}
            sample["scalars"] = sample["scalars"][:, :-1]
            dataloader = T.utils.data.DataLoader(DictDataset(sample), 
                                                 batch_size=64, shuffle=False)
            n_cnts = np.int64(np.ravel(np.concatenate([
                self.flow.sample(i).cpu().detach().numpy() for i in dataloader
                ])))
            eval_ctxt["scalars"][:,-1] = n_cnts
            

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
            #setup paths and folder
            obs_jet_size= self.size #len(eval_ctxt["true_n_cnts"])
            post_save_path = f"{self.eval_folder}/post/"
            os.makedirs(f"{post_save_path}/", exist_ok=True)

            if self.run_flow: # create flow_N folder for flow evals
                post_save_path += "/flow_N/"
                os.makedirs(f"{post_save_path}/", exist_ok=True)
                
            
            # save truth
            self.data.jet_vars.iloc[:obs_jet_size].to_csv(f"{post_save_path}/truth_jets_{saving_name}.csv")
            
            # if combine_ctxt_size==1:
            #     raise ValueError("combine_ctxt_size should be larger than 1")
            
            steps=np.arange(obs_jet_size+combine_ctxt_size, step=combine_ctxt_size)
            for nr, (lower, high) in tqdm(enumerate(zip(steps[:-1], steps[1:])),
                                          total=len(steps)):
                # get ctxt
                ctxt = {i:j[lower:high] for i,j in eval_ctxt.items()}
                # ctxt["cnts"] = np.expand_dims(ctxt["cnts"][ctxt["mask"]],0)
                
                # generate sample
                gen_jet_vars, _, gen_cnts, gen_cnts_mask  = self.generate_post(ctxt, n_post_to_gen)

                # n_post = (gen_cnts_mask*np.r_[np.ones((n_post_to_gen,1))*lower,np.ones((n_post_to_gen,1))*(high-1)])

                # posterior nr
                n_post = gen_cnts_mask*np.repeat(np.arange(lower, high), n_post_to_gen)[:,None]
                
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
                pd.concat(all_gen_jets).to_csv(f"{post_save_path}/gen_jets_{saving_name}.csv")
                
                if "posterior" in saving_name:
                    with h5py.File(f"{post_save_path}/gen_cnts_{saving_name}.h5", 'w') as hf:
                        hf.create_dataset("gen_cnts",  data=np.concatenate(all_gen_cnts, 0))
                else:
                    pd.DataFrame(np.concatenate(all_gen_cnts, 0),
                                columns=["eta", "phi", "pt", "eventNumber", "n_post"]
                                ).to_csv(f"{post_save_path}/gen_cnts_{saving_name}.csv")
        else:
            if "std_0" in saving_name: # run all values for std_0
                self.diffusion.init_noise["size"] = len(eval_ctxt["scalars"])

            n_cnts = eval_ctxt.pop("true_n_cnts")
            initial_noise = ds.generate_gaussian_noise(eval_ctxt=eval_ctxt,
                                                        n_constituents=n_cnts,
                                                        loader_kwargs=self.loader_config,
                                                        **self.diffusion.init_noise)
            all_gen_jets, gen_data_rel, all_gen_cnts, mask = self.generate_sample(initial_noise)
            index = mask*np.arange(len(mask))[:, None]
            all_gen_cnts = pd.DataFrame( np.c_[all_gen_cnts[mask].numpy(),index[mask]],
                                        columns=["eta", "phi", "pt", "eventNumber"])
            jet_size= len(all_gen_jets)
            # save the generated data
            all_gen_jets.to_csv(f"{self.eval_folder}/gen_jets_{saving_name}.csv")
            self.data.jet_vars.iloc[:jet_size].to_csv(f"{self.eval_folder}/truth_jets_{saving_name}.csv")
            all_gen_cnts.to_csv(f"{self.eval_folder}/gen_cnts_{saving_name}.csv")
                
            # TODO truth_cnts should also be added self.data.cnts_vars[:n_points]

        return all_gen_jets, all_gen_cnts


if __name__ == "__main__":
    # %matplotlib widget
    config = hydra_utils.hydra_init(str(root/"configs/evaluate.yaml"))
    print(config)
    
    eval_fw = hydra.utils.instantiate(config.eval)
    save_figs = config.save_figures
    save_path = f"{config.eval.path_to_model}/figures/"

    name = get_pileup_name(eval_fw.data.pileup_dist_args)
    

    gen_data = eval_fw.get_eval_files(should_contain="jets_"+config.csv_sample_to_load.split("_")[0])

    if len(gen_data)==0:
        raise ValueError("Eval files not avaliable - should be generated and saved!")
    
    # calculate number of events
    size = 9999*config.eval.size//10_000
    if ("posterior" in config.csv_sample_to_load) or (config.eval.data_cfg.pileup_dist_args.std==0):
        size = "9999"
        

    truth_jets = gen_data[f"truth_jets_{config.csv_sample_to_load}{name}_size_{size}"]
    mask_mass = truth_jets["mass"]!=0 # some masses are zero?
    truth_jets=truth_jets.reset_index(drop=True)
    logging.info(f"Plotting: gen_jets_{config.csv_sample_to_load}{name}_size_{size}")
    gen_jets = gen_data[f"gen_jets_{config.csv_sample_to_load}{name}_size_{size}"].reset_index(drop=True)
    
    # get ctxt 
    ctxt_path = glob(f'/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/obs_jet{name}*.npy')
    if len(ctxt_path)==0:
        ctxt_path = glob(f'/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/data/obs_jet*.npy')
        
    eval_ctxt= np.load([i for i in ctxt_path if "soft" not in i][0], allow_pickle=True).item()

    obs_jets = pd.DataFrame(eval_ctxt["scalars"][:len(truth_jets)],
                            columns=eval_fw.config.data.train.jet_physics_cfg.jet_scalars_cols).reset_index(drop=True)

    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)

    col_to_plot = config.jet_vars
    jet_labels = [r"$\eta$", r"$\phi$", "Mass [GeV]", r"p$_\mathrm{T}$ [GeV]"]

    softdrop=glob(f"{config.softdrop_path}/softdrop/zcut_0_05_beta_2/*{name}*")


    softdrop_cnts = np.load([i for i in softdrop if "npy" in i][0], allow_pickle=True)[..., [1,2,0]]
    
    softdrop_jet = pd.read_hdf([i for i in softdrop if "HLV" in i][0])

    ### plot figures ##
    if "single" in config.csv_sample_to_load: # single generated value
        save_path=save_path+"single/"
        truth_jets = np.nan_to_num(truth_jets[col_to_plot].values, -999)
        gen_jets = np.nan_to_num(gen_jets[col_to_plot].values, -999)
        softdrop_jet = np.nan_to_num(softdrop_jet[col_to_plot].iloc[:len(truth_jets)].values, -999)
        obs_jets = np.nan_to_num(obs_jets[col_to_plot].iloc[:len(truth_jets)].values, -999)
        
        
        # plot 1d marginals of cnts
        eval_fw.plot_marginals(truth_jets,
                                softdrop_jet,
                                obs_jets,
                                gen_jets,
                                col_name=col_to_plot,
                                hist_kwargs=hist_kwargs,
                                ratio_kwargs=ratio_kwargs,
                                save_path=f"{save_path}/gen_jets_" if save_figs else None,
                                xlabels=jet_labels
                                )
        # sys.exit()

        hist_kwargs["dist_styles"].pop(0)

        hist_kwargs["dist_styles"][0]["label"] =  "SoftDrop"
        hist_kwargs["dist_styles"][1]["label"] = "Obs."
        hist_kwargs["dist_styles"][2]["label"] =  "Vipr"

        diff_obs = ((obs_jets-truth_jets)/truth_jets)[:len(truth_jets)]
        diff_gen = ((gen_jets-truth_jets)/truth_jets)
        diff_SD = ((softdrop_jet[:len(truth_jets)]-truth_jets)/truth_jets)
        
        
        diff_obs = np.nan_to_num(diff_obs, -999)
        diff_gen = np.nan_to_num(diff_gen, -999)
        diff_SD = np.nan_to_num(diff_SD, -999)
        
        hist_kwargs["percentile_lst"]=[5, 95]  # for eta/phi
        args = (diff_SD, diff_obs, diff_gen)
        
        eval_fw.plot_marginals(*args,
                                col_name=col_to_plot[:2],
                                hist_kwargs=hist_kwargs,
                                save_path=f"{save_path}/diff_jets_" if save_figs else None,
                                ratio_bool=False,
                                black_line_bool=True,
                                xlabels=["Relative error of "+ i.replace(" [GeV]", "")for i in jet_labels[:2]])

        hist_kwargs["percentile_lst"]=[2.5, 97.5] # for pt/mass
        
        eval_fw.plot_marginals(*args,
                                col_name=col_to_plot[2:],
                                hist_kwargs=hist_kwargs,
                                save_path=f"{save_path}/diff_jets_" if save_figs else None,
                                ratio_bool=False,
                                black_line_bool=True,
                                xlabels=["Relative error of "+ i.replace(" [GeV]", "")for i in jet_labels[2:]])

        #### plot #-leading cnts
        # truth_index_sort = np.argsort(eval_fw.data.cnts_vars[...,-1],-1)[...,::-1]
        # # truth_cnts_sorted = eval_fw.data.cnts_vars[np.argsort(eval_fw.data.cnts_vars[...,-1],-1)]
        # gen_index_sort = np.argsort(gen_cnts[...,-1],-1)[...,::-1]
        if False: # # leading cnts

            # hist_kwargs["dist_styles"] = [{"label": r"cnts$_{Top}}$"},
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
            ax[nr].hist(percentile_dict[i], range=[0,100], bins=10,
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
    if config.plot_images and "single" in config.csv_sample_to_load: # generate a single value per ctxt
        os.makedirs(f"{save_path}/imgs/", exist_ok=True)
        print("need to load diffusion data")
        eval_files = glob('/srv/beegfs/scratch/users/a/algren/trained_networks/diffusion/online/jet_2024_02_12_15_29_33_272016/eval_files/flow_N/jet_flow_2024_03_11_13_29_10_850955/post/flow_N/*')
        gen_data_cnts = eval_fw.get_eval_files(should_contain="mu_200",
                                               eval_files=eval_files,
                                               specific_file=".h5")
        
        gen_cnts = gen_data_cnts['gen_cnts_posterior_2000_pileup_mu_200_std_0_size_9999']

        # create pc
        gen_cnts, mask = matrix_to_point_cloud(gen_cnts[:, :3],gen_cnts[:, 3],
                                            #   num_per_event_max=max_cnts
                                                )

        # gen_cnts, mask = matrix_to_point_cloud(gen_cnts[["eta", "phi", "pt"]].values,
        #                                         gen_cnts["eventNumber"].values,
        #                                     #   num_per_event_max=max_cnts
        #                                         )
        # get truth
        eval_truth= np.load('/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/data/top_jet.npy',
                            allow_pickle=True).item()

        eval_truth = {
            "cnts": eval_fw.data.cnts_vars,
            # "cnts": eval_fw.data.cnts_vars_rel,
            "mask": eval_fw.data.mask_cnts}


        style={"range":[[-2.5, 2.5], [-np.pi, np.pi]], "bins":64}

        mask_sd = np.all(softdrop_cnts!=0, -1)
        softdrop_cnts_rel = phy.relative_pos(softdrop_cnts[:100],
                                    softdrop_jet[:100],
                                    mask=mask_sd[:100],
                                    reverse=False)

        ctxt_cnts_rel = phy.relative_pos(eval_ctxt["cnts"][:100],
                                    # truth_jets[:100],
                                    eval_ctxt["scalars"][:100],
                                    mask=eval_ctxt["mask"][:100],
                                    reverse=False)

        gen_cnts_rel = phy.relative_pos(gen_cnts[:100],
                                    gen_jets[:100],
                                    mask=mask[:100],
                                    reverse=False)
        # ctxt_cnts = eval_ctxt["cnts"][:100]

        # truth_cnts = phy.relative_pos(eval_truth["cnts"][:100],
        #                             # truth_jets[:100],
        #                             eval_ctxt["scalars"][:100],
        #                             mask=eval_truth["mask"][:100],
        #                             reverse=True)


        # ctxt_images = np.clip(np.log(pc_2_image(eval_ctxt["cnts"][:100], style)+1), 0, 1)
        # scatterplot

        n=2
        style_truth = {"facecolors":'none', "edgecolors":'black', "linewidth":2,
                       "label": "Ground truth",
                       "alpha": 1}
        import matplotlib.lines as mlines
        eval_mask = eval_ctxt["mask"]
        min_vals = eval_ctxt["cnts"][0].min(0)
        max_vals = eval_ctxt["cnts"][0].max(0)
        for sample, mask_, style, name, idx_vals in zip(
            # [ctxt_cnts_rel, softdrop_cnts_rel, gen_cnts_rel],
            [eval_ctxt["cnts"], softdrop_cnts[:100], gen_cnts[:100]],
            [eval_mask, mask_sd, mask],
            [{"color": "red", "label": "Obs. jet"},
            {"color": "green", "label": "SoftDrop"},
            {"color": "blue", "label": "VIPR"},
            ],
            ["obs", "sd", "vipr"],
            [1,1,8]
            ):
            
            for idx in range(idx_vals):
                fig, ax = plt.subplots(1,1, figsize=(8,8))
                
                sample[idx, :, 1] = phy.rescale_phi(sample[idx, :, 1])
                
                ax.scatter(sample[idx, :, 0][mask_[idx]],
                            sample[idx, :, 1][mask_[idx]],
                            s=sample[idx, :, 2][mask_[idx]]*n,
                            # s=np.exp(sample[idx, :, 2][mask_[idx]])*n,
                            **style
                            )

                ax.scatter(eval_truth["cnts"][0, :, 0][eval_truth["mask"][0]],
                        eval_truth["cnts"][0, :, 1][eval_truth["mask"][0]],
                            s=eval_truth["cnts"][0, :, 2][eval_truth["mask"][0]]*n,
                            # s=np.exp(eval_truth["cnts"][idx, :, 2][eval_truth["mask"][idx]])*n,
                            **style_truth
                )

                # Create a legend for the circle sizes
                markers = []
                for circle_size in [np.sqrt(10), np.sqrt(100)]:  # Adjust sizes as needed
                    markers.append(mlines.Line2D([], [], color='black',
                                                 marker='.', linestyle='None',   
                                                markersize=circle_size*n, label=f"{str(int(circle_size**2))} [GeV]"))
                # # Create a new axes object in the same location as the original
                ax2 = ax.twinx()
                ax2.axis('off')

                ax2.legend(handles=markers,#bbox_to_anchor=(0.5, 1.1),
                            loc='lower left',
                            # title=r"p$_T$ scaling [GeV]",
                            frameon=False#, ncol=len(markers)
                            )

                plt.tight_layout()

                ax.set_ylim(min_vals[1], max_vals[1])
                ax.set_xlim(min_vals[0], max_vals[0])
                ax.set_xlabel(r"$\eta$")
                ax.set_ylabel(r"$\phi$")
                ax.legend(frameon=False, loc="upper left")
                if save_figs:
                    misc.save_fig(fig, f"{save_path}/imgs/scatter_images_{name}_{idx}.pdf")
