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
import src.eval_utils as eutils
from plot_eval import get_pileup_name
from sklearn.calibration import calibration_curve
# substructure variables
jet_labels = [r"$\tau_{21}$", r"$\tau_{32}$", r"$\sqrt{\mathrm{d}_{12}}$", 
                r"$\sqrt{\mathrm{d}_{23}}$", r"D$_{2}$", "Mass", r"$p_\mathrm{T}$"]


if __name__ == "__main__":
    plt.rcParams['font.size'] = 20

    # general setup
    config = hydra_utils.hydra_init(str(root/"configs/evaluate.yaml"))
    

    eval_fw = hydra.utils.instantiate(config.eval)

    #names of jet substructures
    jet_vars = config.jet_sub_vars
    
    csv_sample_to_load = "posterior_2000"
    
    file_type = ".h5"
    
    # setup plotting style
    hist_kwargs = OmegaConf.to_object(config.hist_kwargs)
    ratio_kwargs = OmegaConf.to_object(config.ratio_kwargs)

    # setup save path
    save_path = f"{config.eval.path_to_model}/figures/posteriors/"
    os.makedirs(save_path, exist_ok=True)
    
    
    name = get_pileup_name(eval_fw.data.pileup_dist_args)
    
    # get comparison files
    # vipr eval files for N and p(N)
    # vipr_eval_files = {
    #     "Vipr": f"{eval_fw.path_to_model}/eval_files/post/jet_subs",
    #     # "Vipr(p(N$_{single}$))":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval.flow_path}/post/",
    #     "Vipr(p(N))":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval.flow_path}/post/flow_N/jet_subs/"
    #                    }
    vipr_eval_files = {
        "VIPR":f"{eval_fw.path_to_model}/eval_files/flow_N/{config.eval.flow_path}/post/flow_N/jet_subs/"
                       }

    # # get obs. jet
    # # file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    # file_lists_obs = glob(f"{config.obs_jets_path}/jet_subs/*ctxt*.h5")
    # obs_jet_path = [i for i in file_lists_obs if name in i][0]
    # obs_jets = pd.read_hdf(obs_jet_path)
    
    # get truth
    truth_file = glob(f"{config.eval.path_to_model}/eval_files/jet_subs/*truth*.h5")[0]
    truth = pd.read_hdf(truth_file)
    # truth_file = glob(f"{config.eval.path_to_model}/eval_files/*truth*.csv")[0]
    # truth = pd.read_csv(truth_file)
    truth["eventNumber"] = truth.index
    
    posteriors_dict={}

    # get VIPR for both N and p(N)
    for label, i in vipr_eval_files.items():
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
            ax_inte.text(0.6, 0.40, 'Overconfident', size=20, rotation=0)
            ax_inte.set_xlabel("Nominal coverage")
            ax_inte.set_xlim([0,1])
            ax_inte.set_ylim([0,1])
            ax_inte.set_ylabel("Empirical coverage")
            if config.save_figures:
                misc.save_fig(fig_inte, f"{save_path}/coverage_for_structure_vars_{name}_{nr}.pdf")
            nr+=1

        
        # post_width, x_value_of_width = eutils.get_spread_of_post(generated, truth,
        #                                                     variables=jet_vars,
        #                                                     norm_width=True)
        # eutils.plot_post_spread(post_width, x_value_of_width,
        #                         var_names = jet_vars, bins_wth=5,
        #                         y_axis_percentile=[0,99.5],
        #                         xlabels=jet_labels,
        #                         save_path=save_path if config.save_figures else None,
        #                         )