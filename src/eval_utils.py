import matplotlib.pyplot as plt
import copy
import wandb
import numpy as np
import torch as T

#internal
from src.utils import fig2img
from tools.visualization import general_plotting as plot

class EvaluateFramework:

    def plot_marginals(self, *args, col_name, hist_kwargs={}, ratio_kwargs={}, **kwargs):

        if "normalise" not in hist_kwargs:
            hist_kwargs["normalise"]=len(args)>1

        # run wandb if log is present
        log=kwargs.get("log", False)

        # loop over columns
        for nr, name in enumerate(col_name):
            fig, (ax_1, ax_2) = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(9, 5), sharex="col"
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
        if kwargs.get("wandb_bool", True):
            log={}
            fig = fig2img(fig)
            fig = wandb.Image(fig)
            log[name] = fig
            plt.close()
            return log
        else:
            return (fig, ax)
        
        