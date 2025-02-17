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

from tools.tools import misc, hydra_utils
from tools.tools.visualization import general_plotting as plot
from tools.tools.visualization import plot_utils
import src.eval_utils as eutils
from plot_eval import get_pileup_name
from sklearn.calibration import calibration_curve
from tools.tools.datamodule.prepare_data import matrix_to_point_cloud


def get_width(values):
    return (np.percentile(values, 75, 0)-np.percentile(values, 25, 0))/1.349

def relative_error(pred, truth):
    re = (pred-truth[:len(pred)])/truth[:len(pred)]
    return np.nan_to_num(re,-999, posinf=-999, neginf=-999)

plt.rcParams['font.size'] = 22  # General font size
plt.rcParams['axes.labelsize'] = 22  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 22  # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 22  # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 22  # Font size for legend
figsize=(1*8,1*6)

if __name__ == "__main__":
    # general setup
    config = hydra_utils.hydra_init(str(root/"configs/evaluate.yaml"))

    eval_fw = hydra.utils.instantiate(config[config.predict_str], load_model_bool=False)
    eval_fw_clf = hydra.utils.instantiate(config.eval_clf, load_model_bool=False)

    file_type = ".h5"
    
    # setup plotting style
    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)

    # setup save path
    extra = '_with_dropprob' if 'dp' in config.predict_str else ''
    save_path = f"{eval_fw.path_to_model}/figures/single{extra}/"
    os.makedirs(save_path, exist_ok=True)
    
    # substructure variables
    jet_vars = config.jet_sub_vars
    jet_labels = [r"$\tau_{21}$", r"$\tau_{32}$", r"$\sqrt{\mathrm{d}_{12}}$", 
                  r"$\sqrt{\mathrm{d}_{23}}$", r"D$_{2}$", "Mass", r"$p_\mathrm{T}$"]
    
    name = get_pileup_name(eval_fw.data.pileup_dist_args)
    
    # get comparison files
    # vipr eval files for N and p(N)
    # vipr_eval_files = {"Vipr": f"{eval_fw.path_to_model}/eval_files/",
    #                    "Vipr(p(N))":f"{eval_fw.path_to_model}/eval_files/flow_N/{eval_fw.flow_path}"}
    if 'dp' not in config.predict_str:
        vipr_eval_files = {#"Vipr": f"{eval_fw.path_to_model}/eval_files/",
                        "PuppiML":f"{eval_fw_clf.path_to_model}/eval_files/",
                        "VIPR":f"{eval_fw.path_to_model}/eval_files/flow_N/{eval_fw.flow_path}",
                        }
    else:
        vipr_eval_files = {
                        # "VIPR":f"{eval_fw.path_to_model}/eval_files/",
                        "PuppiML":f"{eval_fw_clf.path_to_model}/eval_files/",
                        "VIPR":f"{eval_fw.path_to_model}/eval_files/flow_N/{eval_fw.flow_name}",
                        }

    # get obs. jet
    file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    obs_jet_path = [i for i in file_lists_obs if name in i][0]
    obs_jets = pd.read_hdf(obs_jet_path)
    obs_jets = np.nan_to_num(obs_jets[jet_vars], -999)

    # get softdrop
    file_list_sd =glob(f"{config.softdrop_path}/softdrop/zcut_0_05_beta_2/*{name}*_HLV*")
    legend_name = r"$z_{\mathrm{cut}} = $0.05,"+'\n'+r"$\beta$ = 2.0"
    hist_kwargs['dist_styles'][2]['label'] += f":\n{legend_name}"
    
    softdrop_jet = pd.read_hdf(file_list_sd[0])
    softdrop_jet = np.nan_to_num(softdrop_jet[jet_vars], -999)
    
    # get truth
    truth_file = glob(f"{config.eval_VIPR.path_to_model}/eval_files/jet_subs/*truth*.h5")[1]
    truth = pd.read_hdf(truth_file)
    truth["eventNumber"] = truth.index 
    truth = np.nan_to_num(truth[jet_vars], -999)
    
    args={"Obs": obs_jets, "Ground truth": truth, "SoftDrop": softdrop_jet}

    # get VIPR for both N and p(N)
    for label, i in vipr_eval_files.items():
        if label == "PuppiML":
            probs_cut = str(config.eval_clf.probs_cut).replace('.', '_')
            path_to_load = glob(f"{i}/jet_subs/*{config.csv_sample_to_load}{name}*{probs_cut}*.h5")
        else:
            path_to_load = glob(f"{i}/jet_subs/*{config.csv_sample_to_load}{name}*.h5")

        generated = pd.read_hdf([i for i in path_to_load
                                 if ("truth" not in i) & ("ctxt" not in i)][0])
        args[label] = np.nan_to_num(generated[jet_vars], -999)

    # hist_kwargs["percentile_lst"]=[0, 100]
    hist_kwargs['legend_kwargs'] = {'frameon': False, 'ncol': 2}

    # plot 1d marginals of cnts
    eval_fw.plot_marginals(*args.values(), col_name=jet_vars,
                            save_path=f"{save_path}/gen_jets_" if config.save_figures else None,
                            xlabels=jet_labels,
                            hist_kwargs=hist_kwargs,
                            ratio_kwargs=ratio_kwargs,
                            figsize=figsize,
                            hist_ylim=3
                            )
    
    hist_kwargs["dist_styles"].pop(1)
    truth = args.pop("Ground truth")

    args = {i: relative_error(j, truth) for i,j in args.items()}

    hist_kwargs['legend_kwargs'] = {'loc': 'upper right', 'frameon': False}
    # jet substructure
    for nr, percentile in enumerate([[0.01,98],[0.01,98],[0.01,98],[0.01,98],[0.01,98],[0.01,98]]):
        # hist_kwargs.pop("style", None)
        # hist_kwargs['style'].pop('bins', None)
        hist_kwargs['style'].pop('range', None)
        eval_fw.plot_marginals(
            *[i[:, nr:nr+1] for i in args.values()],
            col_name=[jet_vars[nr]],
            hist_kwargs=hist_kwargs,
            save_path=f"{save_path}/diff_jets_" if config.save_figures else None,
            ratio_bool=False,
            xlabels=[f"Relative error of {i}" for i in [jet_labels[nr]]],
            black_line_bool=True,
            percentile=percentile,
            legend_kwargs=hist_kwargs['legend_kwargs'],
            figsize=figsize,
            )

    # performance as a function of mu
    if True:

        puppi_path = vipr_eval_files.pop('PuppiML')

        vipr_mu = {i: {"median": [], "width": []} for i in vipr_eval_files}
        obs_mu = {"median": [], "width": []} 
        sd_mu = {}
        puppi_mu = {}
        # mu_lst = [50,60,70,80,90,100,150,200,250,300]
        mu_lst = [50,70,80,90,100,150,200,250,300]
        # mu_lst = [60]
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
            
            # softdrop
            for sp_folder in tqdm(softdrop, total=len(softdrop), leave=False):
                # if 'beta_3' in sp_folder:
                #     continue
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

            puppi=glob(f"{puppi_path}/jet_subs/*")
            # PUPPI
            for puppi_file in tqdm(puppi, total=len(puppi), leave=False):

                if name not in puppi_file:
                    continue

                puppi_hp = puppi_file.split('cut_')[-1].replace('.h5','')

                if puppi_hp not in puppi_mu:
                    puppi_mu[puppi_hp] = {"median": [], "width": [], "path": None}

                puppi_jet = pd.read_hdf(puppi_file).iloc[:len(obs_jets)]

                diff_SD = relative_error(puppi_jet[jet_vars],truth)

                puppi_mu[puppi_hp]["width"].append(get_width(diff_SD)[:, None])
                puppi_mu[puppi_hp]["median"].append(np.median(diff_SD,0)[:, None])
                puppi_mu[puppi_hp]["path"] = puppi_file
        # sys.exit()

        for i in ["width", "median"]:
            for j in puppi_mu:
                puppi_mu[j][i] = np.concatenate(puppi_mu[j][i], 1)

            for j in vipr_mu:
                vipr_mu[j][i] = np.concatenate(vipr_mu[j][i], 1)
            obs_mu[i] = np.concatenate(obs_mu[i], 1)

            for j in sd_mu:
                sd_mu[j][i] = np.concatenate(sd_mu[j][i], 1)

        lw=3
        ls_lst = plot_utils.linestyle_tuple*2
        for benchmarks in [[sd_mu, puppi_mu], [puppi_mu]]:
            for i, name in enumerate(jet_labels):

                fig,ax = plt.subplots(1,1, figsize=figsize)
                fig_m,ax_m = plt.subplots(1,1, figsize=figsize)


                # ax.plot(mu_lst,np.zeros_like(mu_lst),label = "Zero line",
                #          color="black", ls="dotted", lw=3)
                
                # obs jet
                if len(benchmarks)>1:
                    ax_m.plot(mu_lst[: len(obs_mu["median"][i, :])], obs_mu["median"][i, :],
                            label="Obs.", color="red", lw=lw, ls="dotted")
                    ax.plot(mu_lst[: len(obs_mu["width"][i, :])], obs_mu["width"][i, :],
                            label="Obs.", color="red", lw=lw, ls="dotted")
                    additional_name = ''
                else:
                    additional_name = '_compare_puppiml'
                    

                for j, line in zip(vipr_mu, ["solid", "dashed"]):
                    ax_m.plot(mu_lst, vipr_mu[j]["median"][i, :],label=j,
                                color="blue", ls=line, lw=lw)
                    ax.plot(mu_lst, vipr_mu[j]["width"][i, :],label=j,
                            color="blue", ls=line, lw=lw)

                # plot n sd parameters
                if True: # plot the envolope of all sd parameters
                    for dist in benchmarks:
                        for ax_i,key in zip([ax, ax_m], ["width", "median"]):
                            if "zcut_0_05_beta_2" in dist:
                                color='green'
                                pred = dist["zcut_0_05_beta_2"][key]
                                legend_name = r"$z_{\mathrm{cut}} = $0.05"+r" $\beta$=2.0"
                                legend_label='SoftDrop'
                                ax_i.plot(mu_lst, pred[i, :], label=legend_name, color=color,
                                        ls='dashed', lw=lw)
                            else:
                                color = 'orange'
                                pred = dist["0_3"][key]
                                legend_name = "PuppiML cut = 0.3"
                                legend_label='PuppiML'
                                ax_i.plot(mu_lst, pred[i, :], label=legend_name, color=color,
                                        ls='dotted', lw=lw)
                            unc_min = np.min([dist[k][key] for k in dist], 0)
                            unc_max = np.max([dist[k][key] for k in dist], 0)
                            
                            # mean = np.mean([sd_mu[k][key] for k in sd_mu], 0)
                            # ax_i.plot(mu_lst, mean[i, :], label="SoftDrop envelope", color="green")

                            ax_i.fill_between(mu_lst, unc_min[i, :], unc_max[i, :], color=color,alpha=0.3,
                                            label=f"{legend_label} envelope")
                else: # plot all sd parameters
                    # for nr, j in enumerate(sd_mu):
                    #     # unpack beta/z naming
                    #     beta = j.split("beta_")[-1].replace("_", ".")
                    #     zcut = j.split("zcut_")[-1].split("_beta")[0].replace("_", ".")
                    
                    #     legend_name = r"$z_{\mathrm{cut}} = $"+zcut+r" $\beta$= "+beta
                        
                    #     ax_m.plot(mu_lst, sd_mu[j]["median"][i, :],label=f"SD: {legend_name}", color="green",
                    #                 ls=ls_lst[nr][1], lw=lw)
                    #     ax.plot(mu_lst, sd_mu[j]["width"][i, :],label=f"SD: {legend_name}", color="green",
                    #             ls=ls_lst[nr][1], lw=lw)

                    for nr, j in enumerate(puppi_mu):
                        # unpack beta/z naming
                        probs = j.replace('_', '.')
                    
                        legend_name = f"PUPPI cut = {probs}"
                        
                        ax_m.plot(mu_lst, puppi_mu[j]["median"][i, :],label=f"PUPPI: {legend_name}", #color="orange",
                                    ls=ls_lst[nr][1], lw=lw)
                        ax.plot(mu_lst, puppi_mu[j]["width"][i, :],label=f"PUPPI: {legend_name}", #color="orange",
                                ls=ls_lst[nr][1], lw=lw)

                for ax_i,j in zip([ax, ax_m], ["IQR", "Bias"]):
                    # if 'Bias' in j and len(benchmarks)==1:
                    #     loc = "lower right"
                    # else:
                    #     loc = "upper right"
                    ax_i.set_ylabel(f"{j} of RE({name})")
                    ax_i.set_xlabel(r"$\mu$")
                    ax_i.set_xlim([50, 300])
                    ylim=ax_i.get_ylim()
                    top_y = 3.5
                    if 'tau' in jet_vars[i]:
                        top_y = 4.5

                    if j=='IQR':
                        ax_i.set_ylim([0, ylim[1]*top_y])
                    else:
                        ax_i.set_ylim([ylim[0]*top_y, ylim[1]*top_y])

                    ax_i.legend(frameon=False, loc='best')#, bbox_to_anchor=(1.6, 1.2))
                # ax_m.set_ylim([-0.1, 1])
                plt.tight_layout()

                # ax.set_yscale("log")
                if config.save_figures:
                    misc.save_fig(fig_m, f"{save_path}/pileup_func/{jet_vars[i]}_median{additional_name}.pdf")
                    misc.save_fig(fig, f"{save_path}/pileup_func/{jet_vars[i]}_IQR{additional_name}.pdf")
