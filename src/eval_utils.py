import matplotlib.pyplot as plt
import wandb

#internal
from src.utils import fig2img
from tools.visualization import general_plotting as plot
import copy

class EvaluateFramework:

    def plot_marginals(self, *args, col_name, hist_kwargs={}, ratio_kwargs={}, **kwargs):
        log=kwargs.get("log", {})
        for nr, name in enumerate(col_name):
            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
                )
            data_col = [d[:,nr] for d in args]
            counts_dict, _ = plot.plot_hist(*data_col, ax=ax_1,normalise=True,
                                            **copy.deepcopy(hist_kwargs))
            
            plot.plot_ratio(counts_dict, truth_key="dist_0", ax=ax_2,
                            zero_line_unc=True,
                            normalise=len(data_col[0])!=len(data_col[1]),
                            ylim=[0.8, 1.2], **copy.deepcopy(ratio_kwargs))
            ax_2.set_xlabel(name)
            log[f"{name}_hist"] =  wandb.Image(fig2img(fig))
            plt.close(fig)

        return log
            
