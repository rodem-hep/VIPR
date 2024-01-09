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
import src.physics as phy 



if __name__ == "__main__":
    config = misc.load_yaml(str(root/"configs/evaluate.yaml"))
    eval_fw = hydra.utils.instantiate(config.eval)
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
        if "eventNumber" not in gen_jets.columns:
            gen_jets=gen_jets.rename(columns={gen_jets.columns[0]:"eventNumber"})

        # sys.exit()
        # generated
        sjets.dump_hlvs(gen_cnts[:size], mask[:size],
                        out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_{config.csv_sample_to_load}.h5",
                        addi_col={"eventNumber":gen_jets["eventNumber"].values[:size]}
                        )

        # # substruct for true Top
        eval_fw.load_diffusion()
        
        sjets.dump_hlvs(eval_fw.data.cnts_vars[:len(gen_jets)],
                        eval_fw.data.mask_cnts[:len(gen_jets)],
                        out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_truth_{config.csv_sample_to_load}.h5",
                        # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                        )

        if "single" in config.csv_sample_to_load: # substruct for obs. jet
            
            eval_ctxt = eval_fw.data.get_normed_ctxt()
            ctxt_cnts = phy.relative_pos(eval_ctxt["cnts"],eval_ctxt["scalars"],
                                         mask=eval_ctxt["mask"], reverse=True)
            sjets.dump_hlvs(ctxt_cnts[:len(gen_jets)], eval_ctxt["mask"][:len(gen_jets)],
                            out_path=f"{config.eval.path_to_model}/eval_files/jet_substructure_ctxt_{config.csv_sample_to_load}.h5",
                            # addi_col={"eventNumber":truth_jets["eventNumber"].values[:len(gen_jets)]}
                            )
    else:
        
        file_lists = glob(f"{config.eval.path_to_model}/eval_files/*{config.csv_sample_to_load}*.h5")
        truth = pd.read_hdf([i for i in file_lists if "truth" in i][0])
        generated = pd.read_hdf([i for i in file_lists if ("truth" not in i) & ("ctxt" not in i)][0])
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
        
        jet_vars = ["tau_21", "tau_32", "d12", "d23", "d2"]#, "mass", "pt"]
        
        if "posterior" in file_lists[0]:
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

            # jet_vars = ['tau_1', 'tau_2', 'tau_3', 'tau_21', 'tau_32', 'd12', 'd23', 'ecf2',
    #    'ecf3', 'd2']
            
            post_width, x_value_of_width = eutils.get_spread_of_post(generated, truth,
                                                                variables=jet_vars,
                                                                norm_width=True)
            eutils.plot_post_spread(post_width, x_value_of_width,
                                    var_names = jet_vars, bins_wth=10,
                                    y_axis_percentile=[0,99.5])
        else:
            # load obs. jet
            obs_jets = pd.read_hdf([i for i in file_lists if "ctxt" in i][0])

            # remove nan
            truth = np.nan_to_num(truth[jet_vars], -999)
            generated = np.nan_to_num(generated[jet_vars], -999)
            obs_jets = np.nan_to_num(obs_jets[jet_vars], -999)
            
            # plot 1d marginals of cnts
            hist_kwargs["dist_styles"] = [{"label": r"jet$_{Top}}$"}, 
                                        {"label": r"jet$_{diffusion}$"},
                                        {"label": r"jet$_{obs.}$"},
                                        ]
            hist_kwargs["style"]["bins"] = 25

            eval_fw.plot_marginals(truth, generated, obs_jets,
                                    col_name=jet_vars,
                                    hist_kwargs=hist_kwargs,
                                    save_path=f"{save_path}/gen_jets_" if config.save_figures else None)

            hist_kwargs["dist_styles"] = [
                {"label": r"(jet$_{diffusion}$-jet$_{Top}}$)/jet$_{Top}}$"},
                {"label": r"(jet$_{observed}$-jet$_{Top}}$)/jet$_{Top}}$"}
                ]
            hist_kwargs["style"]["bins"] = 40
            
            diff_obs = ((obs_jets-truth)/truth)
            diff_gen = ((generated-truth)/truth)
            eval_fw.plot_marginals(
                np.nan_to_num(diff_gen, -999, posinf=-999, neginf=-999),
                np.nan_to_num(diff_obs, -999, posinf=-999, neginf=-999),
                col_name=jet_vars,
                hist_kwargs=hist_kwargs,
                save_path=f"{save_path}/diff_jets_" if config.save_figures else None,
                ratio_bool=False)
