"evaluate posteriors performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import hydra
from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd
from omegaconf import OmegaConf
import os 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools import misc
from tools.visualization import general_plotting as plot
import src.eval_utils as eutils
from plot_eval import get_pileup_name
from sklearn.calibration import calibration_curve

def get_width(values):
    return (np.percentile(values, 75, 0)-np.percentile(values, 25, 0))/1.349

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]

if __name__ == "__main__":
    config = misc.load_yaml(str(root/"configs/evaluate.yaml"))
    eval_fw = hydra.utils.instantiate(config.eval)
    save_path = f"{config.eval.path_to_model}/figures/"
    size=None #512*4
    file_type = ".h5"
    
    name = ("" if "posterior" in config.csv_sample_to_load
            else get_pileup_name(eval_fw.data.pileup_dist_args))
    


    # if int(name.split("_")[-1])!=0: # TODO should be removed at some point
    softdrop=glob(f"{config.softdrop_path}/*softdrop*hlv*z05*b1*")
    truth_file = glob(f"{config.eval.path_to_model}/eval_files/*truth*{config.csv_sample_to_load}*.h5")[0]
    # else:
    #     truth_file = glob(f"{config.eval.path_to_model}/eval_files/*truth*{file_type}")[0]
    #     softdrop=glob(f"{config.softdrop_path}/data/*{name}*softdrop*HLV*")

    file_lists = glob(f"{config.eval.path_to_model}/eval_files/*{config.csv_sample_to_load}*.h5")

    truth = pd.read_hdf(truth_file)
    generated = pd.read_hdf([i for i in file_lists if ("truth" not in i) & ("ctxt" not in i)][0])
    truth["eventNumber"] = truth.index 
    
    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)
    

    # substructure variables
    jet_vars = ["tau_21", "tau_32", "d12", "d23", "d2", "mass", "pt"]
    jet_labels = [r"$\tau_{21}$", r"$\tau_{32}$", r"d$_{12}$", r"d$_{23}$",
                r"d$_{2}$", "Mass [GeV]", r"p$_T$ [GeV]"]

    if "posterior" in config.csv_sample_to_load:
        save_path=save_path+"posteriors/"
        os.makedirs(save_path, exist_ok=True)
        percentile_dict = eutils.get_percentile(generated, truth,
                                            columns=jet_vars)
        x = np.linspace(0, 0.5, 50)

        fig_inte, ax_inte = plt.subplots(1,1, figsize=(8,6), squeeze=True)
        for nr, i in enumerate(percentile_dict):
            fig, (ax_1, ax_2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col")
            counts, _ = plot.plot_hist(np.random.uniform(0, 100, size=1_000_000), percentile_dict[i],
                                        style={"bins":20, "range": [0,100]},
                                        dist_styles = [{"label": "Uniform", "color": "black", "ls": "dashed"},
                                                        {"label": "Vipr", "color": "blue"}],
                                        weights=[np.ones(1_000_000)/1_000_000,
                                                    np.ones_like(percentile_dict[i])/len(percentile_dict[i])],
                                        ax=ax_1,
                                        legend_kwargs={"title": jet_labels[nr]}
                                        )
            plot.plot_ratio(counts, truth_key="dist_0", ax=ax_2,
                            styles=[{"color": "black"}, {"color": "blue"}],
                            zero_line_unc=True)
            bins = counts["bins"]/100
            counts0 = counts["dist_0"]["counts"][0]
            counts1 = counts["dist_1"]["counts"][0]
            ax_2.set_xlabel("Truth quantiles of posterior")
            ax_1.set_ylabel("Normalised counts")

            if config.save_figures:
                misc.save_fig(fig, f"{save_path}/posterior_quantiles_{i}.pdf")

            # x = np.random.uniform(0, 100, size=len(percentile_dict[i]))
            # x = np.sort(x).cumsum()
            # counts = np.sort(percentile_dict[i]).cumsum()
            quantiles_1 = np.quantile(percentile_dict[i], x)
            quantiles_2 = np.quantile(percentile_dict[i], (1-x))
            ax_inte.plot(
                    # x/x[-1],
                    # counts0.cumsum(),
                    # counts/counts[-1],
                    # counts1.cumsum(),
                    (quantiles_2-quantiles_1)[::-1]/100,
                np.linspace(0, 1, len(x)), #bins[:-1]+np.diff(bins)/2, # 
                    label=jet_labels[nr])

        ax_inte.plot(np.linspace(0, 1, len(counts)),
                    np.linspace(0, 1, len(counts)),
                    ls="dashed", color="black",
                    label="Well-calibrated")
        ax_inte.legend(frameon=False)
        ax_inte.set_xlabel("Nominal coverage")
        ax_inte.set_ylabel("Empirical coverage")
        if config.save_figures:
            misc.save_fig(fig_inte, f"{save_path}/coverage_for_structure_vars.pdf")
        
        post_width, x_value_of_width = eutils.get_spread_of_post(generated, truth,
                                                            variables=jet_vars,
                                                            norm_width=True)
        eutils.plot_post_spread(post_width, x_value_of_width,
                                var_names = jet_vars, bins_wth=10,
                                y_axis_percentile=[0,99.5],
                                xlabels=jet_labels,
                                save_path=save_path
                                )
    else:

        save_path=save_path+"single/"
        os.makedirs(save_path, exist_ok=True)
        # load obs. jet
        softdrop_jet = pd.read_hdf(softdrop[0])
        
        # obs jet path maybe there is none
        obs_jet_path = [i for i in file_lists if "ctxt" in i]

        if ".csv" in file_type and len(obs_jet_path)>0:
            obs_jets = pd.read_csv(obs_jet_path[0])
        elif len(obs_jet_path)>0:
            obs_jets = pd.read_hdf(obs_jet_path[0])
        else:
            obs_jets=None
        # remove nan
        truth = np.nan_to_num(truth[jet_vars], -999)
        generated = np.nan_to_num(generated[jet_vars], -999)
        softdrop_jet = np.nan_to_num(softdrop_jet[jet_vars], -999)
        args=[truth, generated, softdrop_jet]

        if obs_jets is not None:
            obs_jets = np.nan_to_num(obs_jets[jet_vars], -999)
            args.append(obs_jets)

        hist_kwargs["percetile_lst"]=[0, 100]

        # plot 1d marginals of cnts
        eval_fw.plot_marginals(*args, col_name=jet_vars,
                                save_path=f"{save_path}/gen_jets_" if config.save_figures else None,
                                xlabels=jet_labels,
                                hist_kwargs=hist_kwargs,
                                ratio_kwargs=ratio_kwargs,
                                )

        hist_kwargs["dist_styles"] = [
            {"label": "Vipr", "color": "blue"},
            {"label": "Obs.", "color": "red"},
            {"label": "SoftDrop", "color": "green"},
            ]

        min_size = min(len(truth), len(generated), len(softdrop_jet))

        truth = truth[:min_size]
        generated = generated[:min_size]
        softdrop_jet = softdrop_jet[:min_size]
        

        diff_gen = np.nan_to_num(
            ((generated-truth)/truth),
            -999, posinf=-999, neginf=-999)

        diff_SD = np.nan_to_num(
            ((softdrop_jet-truth)/truth),
            -999, posinf=-999, neginf=-999)
        args=[diff_gen, diff_SD]
            
        if obs_jets is not None:
            obs_jets = obs_jets[:min_size]
            diff_obs = np.nan_to_num(
                ((obs_jets-truth)/truth),
                -999, posinf=-999, neginf=-999)
            args.append(obs_jets)

        # jet substructure
        eval_fw.plot_marginals(
            *[i[:, :-2]for i in args],
            col_name=jet_vars[:-2],
            hist_kwargs=hist_kwargs,
            save_path=f"{save_path}/diff_jets_" if config.save_figures else None,
            ratio_bool=False,
            xlabels=[f"Relative error of {i}" for i in jet_labels[:-2]],
            black_line_bool=True,
            sym_percentile=99
            )
        
        # mass/pT
        eval_fw.plot_marginals(
            *[i[:, -2:]for i in args],
            col_name=jet_vars[-2:],
            hist_kwargs=hist_kwargs,
            save_path=f"{save_path}/diff_jets_" if config.save_figures else None,
            ratio_bool=False,
            xlabels=[f"{i} response" for i in jet_labels[-2:]],
            black_line_bool=True,
            sym_percentile=99.99
            )
        sys.exit()

        truth_file = glob(f"{config.eval.path_to_model}/eval_files/*truth*{config.csv_sample_to_load}.h5")[0]
        truth = pd.read_hdf(truth_file)

        # performance as a function of mu
        vipr_mu = {"median": [], "width": []}
        obs_mu = {"median": [], "width": []}
        sd_mu = {}
        mu_lst = [50,60,70,80,90,100,150, 200]

        for mu in mu_lst:
            name = get_pileup_name({"mu": mu, "std": 0})

            file_lists = glob(f"{config.eval.path_to_model}/eval_files/*{config.csv_sample_to_load}{name}.h5")

            generated = pd.read_hdf([i for i in file_lists if ("truth" not in i) & ("ctxt" not in i)][0])
            obs_jets = pd.read_hdf([i for i in file_lists if "ctxt" in i][0])

            diff_obs = np.nan_to_num(
                (obs_jets[jet_vars]-truth[jet_vars]),
                -999, posinf=-999, neginf=-999)

            diff_gen = np.nan_to_num(
                (generated[jet_vars]-truth[jet_vars]),
                -999, posinf=-999, neginf=-999)

            
            # obs 
            obs_mu["width"].append(get_width(diff_obs)[:, None])
            obs_mu["median"].append(np.median(diff_obs,0)[:, None])

            # vipr
            vipr_mu["width"].append(get_width(diff_gen)[:, None])
            vipr_mu["median"].append(np.median(diff_gen,0)[:, None])
            
            # SD - handling multiple sd files
            softdrop=glob(f"{config.softdrop_path}/data/*{name}*softdrop*HLV*")
            for sp_path in softdrop:

                softdrop_jet = pd.read_hdf(sp_path).iloc[:len(obs_jets)]

                diff_SD = np.nan_to_num(
                    (softdrop_jet[jet_vars]-truth[jet_vars]),
                    -999, posinf=-999, neginf=-999)

                sd_name = sp_path.split("softdrop")[-1].split("HLV")[0]
                if sd_name not in sd_mu:
                    sd_mu[sd_name] = {"median": [], "width": []}

                sd_mu[sd_name]["width"].append(get_width(diff_SD)[:, None])
                sd_mu[sd_name]["median"].append(np.median(diff_SD,0)[:, None])
        
        for i in ["width", "median"]:
            vipr_mu[i] = np.concatenate(vipr_mu[i], 1)
            obs_mu[i] = np.concatenate(obs_mu[i], 1)
            for j in sd_mu:
                sd_mu[j][i] = np.concatenate(sd_mu[j][i], 1)


        figsize=(1.3*8,1.3*6)
        for i, name in enumerate(jet_labels):

            fig,ax = plt.subplots(1,1, figsize=figsize)
            fig_m,ax_m = plt.subplots(1,1, figsize=figsize)


            # ax.plot(mu_lst,np.zeros_like(mu_lst),label = "Zero line",
            #          color="black", ls="dotted", lw=3)
            ax_m.plot(mu_lst,np.zeros_like(mu_lst),label = "Zero line",
                     color="black", ls="dotted", lw=3)

            ax_m.plot(mu_lst, obs_mu["median"][i, :],label="Obs. jet", color="red")
            ax.plot(mu_lst, obs_mu["width"][i, :],label="Obs. jet", color="red")

            ax_m.plot(mu_lst, vipr_mu["median"][i, :],label="Vipr jet", color="blue")
            ax.plot(mu_lst, vipr_mu["width"][i, :],label="Vipr jet", color="blue")

            # plot n sd parameters
            for nr, j in enumerate(sd_mu):
                # unpack beta/z naming
                beta = j.split("beta_")[-1][:1]
                zcut = j.split("zcut_")[-1].split("_beta")[0].replace("_", ".")
                
                legend_name = r"$z_{\mathrm{cut}} = $"+zcut+r" $\beta$= "+beta
                
                ax_m.plot(mu_lst, sd_mu[j]["median"][i, :],label=f"SD: {legend_name}", color="green",
                          ls=linestyle_str[nr][0])
                ax.plot(mu_lst, sd_mu[j]["width"][i, :],label=f"SD: {legend_name}", color="green",
                        ls=linestyle_str[nr][0])

            for i,j in zip([ax, ax_m], ["IQR", "Median"]):
                i.legend(frameon=False, loc="best")
                i.set_ylabel(f"{j} of {name}")
                i.set_xlabel(r"$<\mu>$")

            ax.set_yscale("log")
            if config.save_figures:
                misc.save_fig(fig_m, f"{save_path}/pileup_func/{name}_median.pdf")    
                misc.save_fig(fig, f"{save_path}/pileup_func/{name}_IQR.pdf")        
