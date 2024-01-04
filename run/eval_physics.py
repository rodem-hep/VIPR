"evaluate diffusion performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import hydra
from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import pandas as pd

from tools import misc
import src.jet_substructure as sjets
from run.evaluation import hist_kwargs 
from tools.datamodule.prepare_data import matrix_to_point_cloud
from tools.visualization import general_plotting as plot
import src.eval_utils as eutils


if __name__ == "__main__":
    config = misc.load_yaml(str(root/"configs/evaluate.yaml"))
    eval_fw = hydra.utils.instantiate(config.eval)
    save_figs = config.save_figures
    save_path = f"{config.eval.path_to_model}/figures/"
    size=None #512*4

    if False:
        ### generate sample ###
        gen_data = eval_fw.get_eval_files(config.csv_sample_to_load)

        truth_jets = gen_data[f"truth_jets_{config.csv_sample_to_load}"]
        gen_cnts = gen_data[f"gen_cnts_{config.csv_sample_to_load}"]
        gen_jets = gen_data[f"gen_jets_{config.csv_sample_to_load}"]

        # # create pc
        gen_cnts, mask = matrix_to_point_cloud(gen_cnts[["eta", "phi", "pt"]].values,
                                                gen_cnts["eventNumber"].values,
                                            #   num_per_event_max=max_cnts
                                                )
        # sys.exit()
        # generated
        sjets.dump_hlvs(gen_cnts[:size], mask[:size],
                        out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_{config.csv_sample_to_load}.h5",
                        addi_col={"eventNumber":gen_jets["eventNumber"].values[:size]}
                        )

        # truth
        eval_fw.load_diffusion()
        
        sjets.dump_hlvs(eval_fw.data.cnts_vars[:len(gen_jets)],
                        eval_fw.data.mask_cnts[:len(gen_jets)],
                        out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_truth_{config.csv_sample_to_load}.h5",
                        # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                        )
    else:
        
        file_lists = glob(f"{config.eval.path_to_model}/eval_files/*{config.csv_sample_to_load}*.h5")
        truth = pd.read_hdf([i for i in file_lists if "truth" in i][0])
        generated = pd.read_hdf([i for i in file_lists if "truth" not in i][0])
        truth["eventNumber"] = truth.index

        # hist_kwargs["style"]["bins"]=25
        # for col in ["tau_21", "tau_32", "d12", "d23", "d2", "mass"]:
        #     fig, (ax_1, ax_2) = plt.subplots(
        #         2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
        #         )
        #     counts, _ = plot.plot_hist(np.nan_to_num(truth[col][:], -999), generated[col][:], ax=ax_1, **deepcopy(hist_kwargs))
        #     plot.plot_ratio(counts, truth_key="dist_0", ax=ax_2, ylim=[0.9, 1.1])
        #     ax_2.set_xlabel(col)
        #     misc.save_fig(fig, f"{save_path}/{col}.png")
        
        jet_vars = ["tau_21", "tau_32", "d12", "d23", "d2", "mass", "pt"]
        percentile_dict = eutils.get_percentile(generated, truth,
                                            columns=jet_vars)
            
        fig_inte, ax_inte = plt.subplots(1,1, figsize=(8,6), squeeze=True)
        for nr, i in enumerate(percentile_dict):
            fig, ax = plt.subplots(1,1, figsize=(8,6), squeeze=True)
            counts, bins, _, = ax.hist(percentile_dict[i], range=[0,100], bins=50,
                     label="Posterior")
            ax.set_xlabel(i)
            plt.legend()
            if config.save_figures:
                misc.save_fig(fig, f"{save_path}/posterior_{i}.pdf")

            ax_inte.plot(np.linspace(0, 1, len(counts)), 
                     counts.cumsum()/len(percentile_dict[i]),
                     label=i)
        ax_inte.plot(np.linspace(0, 1, len(counts)),
                    np.linspace(0, 1, len(counts)),
                    ls="dashed", color="black",
                    label="Uniform")
        ax_inte.legend(frameon=False)
        ax_inte.set_xlabel("Nominal coverage")
        ax_inte.set_ylabel("Empirical coverage")
        if config.save_figures:
            misc.save_fig(fig_inte, f"{save_path}/coverage_for_structure_vars.pdf")

        jet_vars = ["pt", "mass"]
        
        post_width, x_value_of_width = eutils.get_spread_of_post(generated, truth,
                                                               variables=jet_vars,
                                                               norm_width=True)
        eutils.plot_post_spread(post_width, x_value_of_width,
                                var_names = jet_vars, bins_wth=10,
                                y_axis_percentile=[0,99.5])

