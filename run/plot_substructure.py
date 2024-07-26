"evaluate posteriors performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import hydra
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import os 
from tqdm import tqdm

from tools import misc, hydra_utils
from tools.visualization import general_plotting as plot
from tools.visualization import plot_utils
import src.eval_utils as eutils
from plot_eval import get_pileup_name
from sklearn.calibration import calibration_curve

def get_width(values):
    return (np.percentile(values, 75, 0)-np.percentile(values, 25, 0))/1.349

def relative_error(pred, truth):
    re = (pred-truth[:len(pred)])/truth[:len(pred)]
    return np.nan_to_num(re,-999, posinf=-999, neginf=-999)

if __name__ == "__main__":
    plt.rcParams['font.size'] = 20
    # general setup
    config = hydra_utils.hydra_init(str(root/"configs/evaluate.yaml"))

    eval_fw = hydra.utils.instantiate(config.eval)

    file_type = ".h5"
    
    # setup plotting style
    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)

    # setup save path
    save_path = f"{config.eval.path_to_model}/figures/single/"
    os.makedirs(save_path, exist_ok=True)
    
    # substructure variables
    jet_vars = config.jet_sub_vars
    jet_labels = [r"$\tau_{21}$", r"$\tau_{32}$", r"d$_{12}$", r"d$_{23}$",
                r"d$_{2}$", "Mass", r"$p_\mathrm{T}$"]
    
    name = get_pileup_name(eval_fw.data.pileup_dist_args)
    
    # get comparison files
    # vipr eval files for N and p(N)
    # vipr_eval_files = {"Vipr": f"{eval_fw.path_to_model}/eval_files/",
    #                    "Vipr(p(N))":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval.flow_path}"}
    vipr_eval_files = {#"Vipr": f"{eval_fw.path_to_model}/eval_files/",
                       "VIPR":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval.flow_path}"}

    # get obs. jet
    file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    obs_jet_path = [i for i in file_lists_obs if name in i][0]
    obs_jets = pd.read_hdf(obs_jet_path)
    obs_jets = np.nan_to_num(obs_jets[jet_vars], -999)
    
    # get softdrop
    file_list_sd =glob(f"{config.softdrop_path}/softdrop/zcut_0_05_beta_2/*{name}*{file_type}")
    legend_name = r"$z_{\mathrm{cut}} = $0.05,"+'\n'+r"$\beta$ = 2.0"
    hist_kwargs['dist_styles'][1]['label'] += f":\n{legend_name}"
    
    softdrop_jet = pd.read_hdf(file_list_sd[0])
    softdrop_jet = np.nan_to_num(softdrop_jet[jet_vars], -999)
    
    # get truth
    truth_file = glob(f"{config.eval.path_to_model}/eval_files/jet_subs/*truth*.h5")[0]
    truth = pd.read_hdf(truth_file)
    truth["eventNumber"] = truth.index 
    truth = np.nan_to_num(truth[jet_vars], -999)
    
    args={"Top": truth, "SoftDrop": softdrop_jet, "Obs": obs_jets}

    # get VIPR for both N and p(N)
    for label, i in vipr_eval_files.items():
        path_to_load = glob(f"{i}/jet_subs/*{config.csv_sample_to_load}{name}*.h5")

        generated = pd.read_hdf([i for i in path_to_load
                                 if ("truth" not in i) & ("ctxt" not in i)][0])
        args[label] = np.nan_to_num(generated[jet_vars], -999)

    hist_kwargs["percetile_lst"]=[0, 100]

    # plot 1d marginals of cnts
    eval_fw.plot_marginals(*args.values(), col_name=jet_vars,
                            save_path=f"{save_path}/gen_jets_" if config.save_figures else None,
                            xlabels=jet_labels,
                            hist_kwargs=hist_kwargs,
                            ratio_kwargs=ratio_kwargs,
                            )
    hist_kwargs["dist_styles"] = hist_kwargs["dist_styles"][1:]
    
    truth = args.pop("Top")
    
    args = {i: relative_error(j, truth) for i,j in args.items()}

    hist_kwargs['legend_kwargs'] = {'loc': 'upper left', 'frameon': False}
    # jet substructure
    eval_fw.plot_marginals(
        *args.values(),
        col_name=jet_vars,
        hist_kwargs=hist_kwargs,
        save_path=f"{save_path}/diff_jets_" if config.save_figures else None,
        ratio_bool=False,
        xlabels=[f"Relative error of {i}" for i in jet_labels],
        black_line_bool=True,
        sym_percentile=95,
        legend_kwargs=hist_kwargs['legend_kwargs'],
        )

    # performance as a function of mu
    vipr_mu = {i: {"median": [], "width": []} for i in vipr_eval_files}
    obs_mu = {"median": [], "width": []} 
    sd_mu = {}
    mu_lst = [50,60,70,80,90,100,150,200,250,300]
    os.makedirs(f"{save_path}/pileup_func/", exist_ok=True)

    for mu in tqdm(mu_lst):
        name = get_pileup_name({"mu": mu, "std": 0})

        # obs jet
        obs_jet_path = [i for i in file_lists_obs if name in i]
        if len(obs_jet_path)>0:
            obs_jets = pd.read_hdf(obs_jet_path[0])
            
            # relative error
            diff_obs = relative_error(obs_jets[jet_vars].values,truth)

            # get median and width
            obs_mu["width"].append(get_width(diff_obs)[:, None])
            obs_mu["median"].append(np.median(diff_obs,0)[:, None])

        # vipr
        for label, i in vipr_eval_files.items():
            path_to_load = glob(f"{i}/jet_subs/*{config.csv_sample_to_load}{name}*.h5")

            generated = pd.read_hdf([i for i in path_to_load
                                    if ("truth" not in i) & ("ctxt" not in i)][0])
        
            diff_gen = relative_error(generated[jet_vars], truth)
            vipr_mu[label]["width"].append(get_width(diff_gen)[:, None])
            vipr_mu[label]["median"].append(np.median(diff_gen,0)[:, None])
        
        # SD - handling multiple sd files
        softdrop=glob(f"{config.softdrop_path}/softdrop/*")
        for sp_folder in softdrop:

            sp_hp = sp_folder.split("/")[-1]
            if sp_hp not in sd_mu:
                sd_mu[sp_hp] = {"median": [], "width": [],
                                "path": None}

            for sp_path in glob(f"{sp_folder}/*{name}*HLV*"):
                softdrop_jet = pd.read_hdf(sp_path).iloc[:len(obs_jets)]

                diff_SD = relative_error(softdrop_jet[jet_vars],truth)

                sd_mu[sp_hp]["width"].append(get_width(diff_SD)[:, None])
                sd_mu[sp_hp]["median"].append(np.median(diff_SD,0)[:, None])
                sd_mu[sp_hp]["path"] = sp_path

    for i in ["width", "median"]:
        for j in vipr_mu:
            vipr_mu[j][i] = np.concatenate(vipr_mu[j][i], 1)
        obs_mu[i] = np.concatenate(obs_mu[i], 1)
        for j in sd_mu:
            sd_mu[j][i] = np.concatenate(sd_mu[j][i], 1)


    figsize=(1.5*8,1.5*6)
    for i, name in enumerate(jet_labels):

        fig,ax = plt.subplots(1,1, figsize=figsize)
        fig_m,ax_m = plt.subplots(1,1, figsize=figsize)


        # ax.plot(mu_lst,np.zeros_like(mu_lst),label = "Zero line",
        #          color="black", ls="dotted", lw=3)

        ax_m.plot(mu_lst[: len(obs_mu["median"][i, :])], obs_mu["median"][i, :],label="Obs.", color="red")
        ax.plot(mu_lst[: len(obs_mu["width"][i, :])], obs_mu["width"][i, :],label="Obs.", color="red")
        for j, line in zip(vipr_mu, ["solid", "dashed"]):
            ax_m.plot(mu_lst, vipr_mu[j]["median"][i, :],label=j,
                        color="blue", ls=line)
            ax.plot(mu_lst, vipr_mu[j]["width"][i, :],label=j,
                    color="blue", ls=line)


        # plot n sd parameters
        if False: # plot the envolope of all sd parameters
            for ax_i,key in zip([ax, ax_m], ["width", "median"]):
                unc_min = np.min([sd_mu[k][key] for k in sd_mu], 0)
                unc_max = np.max([sd_mu[k][key] for k in sd_mu], 0)
                # mean = np.mean([sd_mu[k][key] for k in sd_mu], 0)
                # ax_i.plot(mu_lst, mean[i, :], label="SoftDrop envelope", color="green")
                ax_i.fill_between(mu_lst, unc_min[i, :], unc_max[i, :], color="green",alpha=0.3,
                                  label="SoftDrop envelope")
                
                pred = sd_mu["zcut_0_05_beta_2"][key]
                legend_name = r"$z_{\mathrm{cut}} = $0.05"+r" $\beta$=2.0"
                ax_i.plot(mu_lst, pred[i, :], label=legend_name, color="green",
                          ls='dashed')
        # else: # plot all sd parameters
        else: # plot all sd parameters
            for nr, j in enumerate(sd_mu):
                # unpack beta/z naming
                beta = j.split("beta_")[-1].replace("_", ".")
                zcut = j.split("zcut_")[-1].split("_beta")[0].replace("_", ".")
            
                legend_name = r"$z_{\mathrm{cut}} = $"+zcut+r" $\beta$= "+beta
                
                ax_m.plot(mu_lst, sd_mu[j]["median"][i, :],label=f"SD: {legend_name}", color="green",
                            ls=plot_utils.linestyle_tuple[nr][1])
                ax.plot(mu_lst, sd_mu[j]["width"][i, :],label=f"SD: {legend_name}", color="green",
                        ls=plot_utils.linestyle_tuple[nr][1])
        

        for ax_i,j in zip([ax, ax_m], ["IQR", "Bias"]):
            ax_i.legend(frameon=False, loc="upper left")
            ax_i.set_ylabel(f"{j} of RE({name})")
            ax_i.set_xlabel(r"$\mu$")
            ax_i.set_xlim([50, 300])
        # ax.set_ylim([0, 1])
        # ax_m.set_ylim([-0.1, 1])

        # ax.set_yscale("log")
        if config.save_figures:
            misc.save_fig(fig_m, f"{save_path}/pileup_func/{jet_vars[i]}_median.pdf")
            misc.save_fig(fig, f"{save_path}/pileup_func/{jet_vars[i]}_IQR.pdf")
