import matplotlib.pyplot as plt
import copy
import wandb
import numpy as np
import torch as T
import scipy.stats as stats
from tqdm import tqdm

#internal
from src.utils import fig2img
import tools.misc as misc
from tools.visualization import general_plotting as plot

def get_percentile(gen_jet_vars, truth_jet_vars, columns):
    percentile_dict = {i:[] for i in columns}
    for i in np.unique(gen_jet_vars.eventNumber):
        for key in percentile_dict:
            mask_evt = gen_jet_vars["eventNumber"]==i
            percentile_dict[key].append(
                stats.percentileofscore(gen_jet_vars[key][mask_evt],
                                        truth_jet_vars[key].iloc[int(i)])
                )
    return percentile_dict

class EvaluateFramework:
    # def __init__()
    
    def __call__(*args, **kwargs):
        print(args)

    def plot_marginals(self, *args, col_name, hist_kwargs={}, ratio_kwargs={}, **kwargs):

        if "normalise" not in hist_kwargs:
            hist_kwargs["normalise"]=len(args)>1

        # run wandb if log is present
        log=kwargs.get("log", False) # not used any more - it was log transform on variables

        # loop over columns
        for nr, name in enumerate(col_name):
            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex="col"
                )

            # unpack data
            data_col = [d[:,nr] for d in args]

            # plot ratio between distribution
            counts_dict, _ = plot.plot_hist(*data_col, ax=ax_1,
                                            **copy.deepcopy(hist_kwargs))

            if len(args)>1:
                # plot ratio between distribution
                plot.plot_ratio(counts_dict, truth_key="dist_0", ax=ax_2,
                                zero_line_unc=True,
                                normalise=len(data_col[0])!=len(data_col[1]),
                                ylim=[0.8, 1.2], **copy.deepcopy(ratio_kwargs))
            ax_2.set_xlabel(name)

            if isinstance(log, dict):
                log[f"{name}_hist"] =  wandb.Image(fig2img(fig))
                plt.close(fig)
            if kwargs.get("save_path", None) is not None:
                misc.save_fig(fig, f"{kwargs['save_path']}{name}.png")

        return log
            
    def plot_images(self, images, epoch=None, mask=None,
                    num_rows=3, num_cols=3, name="",
                    **kwargs):
        if np.argmin(list(images.shape)) !=3:
            raise ValueError("Dimensions of images are incorrect")
            # images= images.permute(0, 2, 3, 1)

        if isinstance(images, T.Tensor):
            images = images.cpu().detach().numpy()
            
        if isinstance(mask, T.Tensor):
            mask = mask.cpu().detach().numpy()

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(8*num_rows, 6*num_cols))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                    
                if len(images.shape)==3:
                    ax[row, col].scatter(images[index, mask[index], 0],
                                images[index, mask[index], 1],
                                s=images[index, mask[index], 2],
                                marker="o"
                                )
                elif len(images.shape)==4:
                    style = {"vmin":0, "vmax":1}
                    if images[index].shape[-1]:
                        style["cmap"] = "gray"
                    ax[row, col].imshow(images[index], **style)
                    ax[row, col].axis("off")
                else:
                    raise ValueError("Unknown data type")
        plt.tight_layout()
        plt.title(epoch)

        if kwargs.get("save_path", None) is not None:
            misc.save_fig(fig, f"{kwargs['save_path']}{name}.png")

        if kwargs.get("wandb_bool", True):
            log={}
            fig = fig2img(fig)
            fig = wandb.Image(fig)
            log[name] = fig
            plt.close()
            return log
        else:
            return (fig, ax)

def get_spread_of_post(gen_jets, truth_jets, variables:list, norm_width:bool=True):
    # calculate width of posterior
    post_width={}
    x_value_of_width={}
    
    # loop over variables
    for col in variables:
        post_width[col]=[]
        x_value_of_width[col]=[]
        for i in tqdm(range(int(gen_jets["eventNumber"].max()))):
            mask_evt = gen_jets["eventNumber"]==i
            post_width[col].append(np.std(gen_jets[mask_evt][col]))
            x_value_of_width[col].append(truth_jets[col].iloc[i])
        x_value_of_width[col]=np.array(x_value_of_width[col])
        post_width[col]=np.array(post_width[col])
        if norm_width:
            post_width[col] = post_width[col]/np.array(x_value_of_width[col])

    return post_width, x_value_of_width

def plot_post_spread(post_width, x_value_of_width, var_names:list, bins_wth=10,
                     x_axis_percentile=None, y_axis_percentile=None, save_path=None, **kwargs):
    # calculate spread of posteriors as a function of a variable

    for var_name in var_names:

        width=[]
        mean=[]
        if kwargs.get("percentile_bins") is None:
            bins = np.percentile(x_value_of_width[var_name], np.arange(0, 100+bins_wth, bins_wth))
        else:
            bins = np.linspace(x_value_of_width[var_name].min(),
                               x_value_of_width[var_name].max(),
                               bins_wth)
        for low,high in zip(bins[:-1], bins[1:]):
            mask = (x_value_of_width[var_name]>=low) & (x_value_of_width[var_name]<high)
            width.append(np.percentile(post_width[var_name][mask], [25,75]))
            mean.append(np.mean(post_width[var_name][mask]))

        width = np.array(width)
        mean = np.array(mean)
        
        fig = plt.figure()
        style={"baseline": None, "edges": bins, "color": "red", "lw":1.5}

        plt.scatter(x_value_of_width[var_name],post_width[var_name], color="blue",
                    s=2, label=r"$\sigma$ of posterior")
        plt.stairs(width[:,0], label="Spread", ls="dashed", **style)
        plt.stairs(mean, label="Mean", **style)
        plt.stairs(width[:,1], ls="dashed", **style)
        
        plt.ylabel("Normalised width of posterioirs")
        plt.xlabel(var_name)
        if y_axis_percentile is not None:
            plt.ylim(np.percentile(post_width[var_name], y_axis_percentile))

        if x_axis_percentile is not None:
            plt.ylim(np.percentile(x_value_of_width[var_name], x_axis_percentile))
        plt.legend()
        if save_path is not None:
            misc.save_fig(fig, f"{save_path}/spread_of_posterior_{var_name}.pdf")

