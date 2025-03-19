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
import src.eval_utils as eutils
from plot_eval import get_pileup_name

# substructure variables
jet_labels = [r"$\tau_{21}$", r"$\tau_{32}$", r"$\sqrt{\mathrm{d}_{12}}$", 
                r"$\sqrt{\mathrm{d}_{23}}$", r"D$_{2}$", "Mass", r"$p_\mathrm{T}$"]


if __name__ == "__main__":
    plt.rcParams['font.size'] = 20

    # general setup
    config = hydra_utils.hydra_init(str(root/"configs/evaluate.yaml"))
    

    eval_fw = hydra.utils.instantiate(config.eval_VIPR, load_model_bool=False)
    eval_fw_clf = hydra.utils.instantiate(config.eval_clf, load_model_bool=False)

    #names of jet substructures
    jet_vars = config.jet_sub_vars
    
    csv_sample_to_load = "posterior_2000"
    
    file_type = ".h5"
    
    # setup plotting style
    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)

    # setup save path
    save_path = f"{config.eval_VIPR.path_to_model}/figures/posteriors/"
    os.makedirs(save_path, exist_ok=True)
    
    
    name = get_pileup_name(eval_fw.data.pileup_dist_args)
    
    # get comparison files
    # vipr eval files for N and p(N)
    # vipr_eval_files = {
    #     "Vipr": f"{eval_fw.path_to_model}/eval_files/post/jet_subs",
    #     # "Vipr(p(N$_{single}$))":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval_VIPR.flow_path}/post/",
    #     "Vipr(p(N))":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval_VIPR.flow_path}/post/flow_N/jet_subs/"
    #                    }
    vipr_eval_files = {
        "VIPR":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval_VIPR.flow_path}/post/flow_N/jet_subs/",
        "PuppiML":f"{eval_fw_clf.path_to_model}/eval_files/",

                       }
    
    # get softdrop
    file_list_sd =glob(f"{config.softdrop_path}/softdrop/zcut_0_05_beta_2/*{name}*_HLV*")
    legend_name = r"$z_{\mathrm{cut}} = $0.05,"+'\n'+r"$\beta$ = 2.0"
    
    softdrop_jet = pd.read_hdf(file_list_sd[0])
    softdrop_jet = np.nan_to_num(softdrop_jet[jet_vars], -999)

    # # get obs. jet
    # # file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    # file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    # obs_jet_path = [i for i in file_lists_obs if name in i][0]
    # obs_jets = pd.read_hdf(obs_jet_path)
    
    # get truth
    truth_file = glob(f"{config.eval_VIPR.path_to_model}/eval_files/jet_subs/*truth*.h5")[0]
    truth = pd.read_hdf(truth_file)
    # truth_file = glob(f"{config.eval_VIPR.path_to_model}/eval_files/*truth*.csv")[0]
    # truth = pd.read_csv(truth_file)
    truth["eventNumber"] = truth.index
    
    posteriors_dict={}

    # get VIPR for both N and p(N)
    for label, i in vipr_eval_files.items():
        if label == "PuppiML":
            probs_cut = str(config.eval_clf.probs_cut).replace('.', '_')
            path_to_load = glob(f"{i}/jet_subs/*{config.csv_sample_to_load}{name}*{probs_cut}*.h5")[0]
        else:
            path_to_load = glob(f"{i}/*jet*{csv_sample_to_load}*{name}*")
            path_to_load = [i for i in path_to_load
                            if ("truth" not in i) & ("ctxt" not in i)][0]

        if ".h5" in path_to_load:
            generated = pd.read_hdf(path_to_load)
        else:
            generated = pd.read_csv(path_to_load)

        if "eventNumber" not in generated:
            generated["eventNumber"] = np.repeat(np.arange(len(generated)//512),512)
        posteriors_dict[label] = generated
    
    puppiml = posteriors_dict.pop('PuppiML')

    percentile_dict = {}
    for name,generated in posteriors_dict.items():
        percentile_dict[name] = pd.DataFrame.from_dict(
            eutils.get_percentile(generated, truth, columns=jet_vars))
        print(percentile_dict[name].shape)
    x = np.linspace(0, 0.5, 50)

    quantiles = {}
    quantiles = {i:{}for i in percentile_dict}
    for label, label_name in zip(jet_vars, jet_labels):
        fig, (ax_1, ax_2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col")
        uni_d_s= {"label": "Uniform", "color": "black", "ls": "dashed"}
        for nr, (name, percentile_vals) in enumerate(percentile_dict.items()):
            if label not in quantiles[name]:
                quantiles[name][label] = []
            
            vals = percentile_vals[label]
            counts, _ = plot.plot_hist(np.random.uniform(0, 100, size=10_000_000), vals,
                                        style={"bins":10, "range": [0,100]},
                                        dist_styles = [uni_d_s,
                                                       {"label": name, "color": 'blue'}],
                                        weights=[np.ones(10_000_000)/10_000_000,
                                                 np.ones_like(vals)/len(vals)],
                                        ax=ax_1,
                                        legend_kwargs={"title": label_name,"prop":{'size': 22},
                                                       "title_fontsize":22}
                                        )
            plot.plot_ratio(counts, truth_key="dist_0", ax=ax_2,
                            styles=[{"color": "black"}, {"color": 'blue'}], ylim=[0.5, 1.5],
                            zero_line_unc=True)
            bins = counts["bins"]/100
            counts0 = counts["dist_0"]["counts"][0]
            counts1 = counts["dist_1"]["counts"][0]
            ax_2.set_xlabel("Truth quantiles of posterior")
            ax_1.set_ylabel("Normalised counts")
            uni_d_s.pop("label", None)

            vals = vals[~np.isnan(vals)]
            quantiles_1 = np.quantile(vals, x)
            quantiles_2 = np.quantile(vals, (1-x))
            quantiles[name][label] = (quantiles_2-quantiles_1)[::-1]/100
        if config.save_figures:
            misc.save_fig(fig, f"{save_path}/posterior_quantiles_{label}.pdf")

    nr=0
    for jet_var, jet_label in zip([jet_vars[:4], jet_vars[4:]],
                       [jet_labels[:4], jet_labels[4:]]):
        for name, quan in quantiles.items():
            fig_inte, ax_inte = plt.subplots(1,1, figsize=(8,6), squeeze=True)
            quan = {i: quan[i] for i in jet_var}
            for (var, vals), label in zip(quan.items(), jet_label):
                ax_inte.plot(vals, np.linspace(0, 1, len(x)), label=label)

            ax_inte.plot(np.linspace(0, 1, len(counts)),
                        np.linspace(0, 1, len(counts)),
                        ls="dashed", color="black")

            ax_inte.legend(frameon=False, title=name)
            ax_inte.text(0.3, 0.8, 'Underconfident', size=20, rotation=0)
            ax_inte.text(0.2, 0.10, 'Overconfident', size=20, rotation=0)
            ax_inte.set_xlabel("Nominal coverage")
            ax_inte.set_xlim([0,1])
            ax_inte.set_ylim([0,1])
            ax_inte.set_ylabel("Empirical coverage")
            if config.save_figures:
                misc.save_fig(fig_inte, f"{save_path}/coverage_for_structure_vars_{name}_{nr}.pdf")
            nr+=1

        # example of a posterior
        dist_styles = config.hist_kwargs['dist_styles']
        for nr in range(2):
            vals = posteriors_dict['VIPR'][posteriors_dict['VIPR']['eventNumber']==nr]
            for _nr, i in enumerate(jet_vars):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # plot ratio between distribution
                counts_dict, _ = plot.plot_hist(vals[i].values, ax=ax,
                                                dist_styles=[{"label": "VIPR", "color": "blue"},],
                                                style={'bins': 20})
                height = np.max(counts_dict['dist_0']['counts'])
                ax.axvline(truth.iloc[nr][i], label="Ground truth",
                           ymax=height, color="black")
                ax.axvline(softdrop_jet[nr,_nr], ymax=height, **dist_styles[2])
                ax.axvline(puppiml.iloc[nr][i],ymax=height, **dist_styles[-2])
                ax.set_xlabel(jet_labels[_nr])
                plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2,
                           columnspacing=1)
                plt.tight_layout()
                current_ylim = ax.get_ylim()
                ax.set_ylim([0, current_ylim[-1]*1.2])

                if config.save_figures:
                    misc.save_fig(fig, f"{save_path}/single_posteriors/{i}_{nr}.pdf")

        
        # post_width, x_value_of_width = eutils.get_spread_of_post(generated, truth,
        #                                                     variables=jet_vars,
        #                                                     norm_width=True)
        # eutils.plot_post_spread(post_width, x_value_of_width,
        #                         var_names = jet_vars, bins_wth=5,
        #                         y_axis_percentile=[0,99.5],
        #                         xlabels=jet_labels,
        #                         save_path=save_path if config.save_figures else None,
        #                         )