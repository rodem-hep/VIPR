"evaluate diffusion performance"
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
from glob import glob
from tools.visualization import general_plotting as plot
import matplotlib.pyplot as plt
import os

import hydra
from copy import deepcopy
import torch as T
import numpy as np
from tqdm import tqdm

from src.eval_utils import get_percentile
from run.run_flow import load_flow_from_path

from tools import misc
import tools.visualization.general_plotting as plot
from tools.physics import relative_pos

class DictDataset(T.utils.data.Dataset):
    def __init__(self, data:dict):
        self.data = data
        self.keys = list(data.keys())

    def __getitem__(self, index):
        return {i:self.data[i][index] for i in self.keys}
    
    def __len__(self):
        return len(self.data[self.keys[0]])

if __name__ == "__main__":
    # %matplotlib widget
    config = misc.load_yaml(str(root/"configs/eval_flow.yaml"))
    obs_jets_paths = glob(f"{config.data_path}/*")
    
    for mu in [200]:
    # for mu in tqdm([50,60,70,80,90,100,150,200,250,300]):
        
        pileup_name = f"mu_{mu}_std_{config.pileup_cfg.std}"
        
        # load data
        obs_jets_path = [i for i in obs_jets_paths if pileup_name in i][0]
        obs_jets = np.load(obs_jets_path, allow_pickle=True).item()
        
        # prepare data for flow
        eval_ctxt = deepcopy(obs_jets)
        truth_cnsts = eval_ctxt.pop("true_n_cnts")
        eval_ctxt["scalars"] = eval_ctxt["scalars"][:, :-1]

        eval_ctxt["cnts"] = relative_pos(eval_ctxt["cnts"], eval_ctxt["scalars"][:, :3],
                                        eval_ctxt["mask"])

        dataloader = T.utils.data.DataLoader(DictDataset(eval_ctxt), batch_size=80,
                                             shuffle=False)
        
        dev = "cuda" if T.cuda.is_available() else "cpu"

        n_iters = 1000
        for name, path in config.path_lst.items():
            flow = load_flow_from_path(path, dev)
            
            pred_n=[]
            for nr, sample in tqdm(enumerate(dataloader), total=n_iters):
                sample = {i:j.to(dev) for i,j in sample.items()}
                pred_n.append(flow.sample(sample, 1).cpu().detach().numpy())
                if nr==n_iters:
                    break
            pred_n= np.concatenate(pred_n, axis=0)
            
            # eiter save the new N counts or plot and compare to the truth
            if config.save_new_N_cnts:
                save_path = f'{config.data_path}/flow_N/{path.split("/")[-1]}'
                os.makedirs(save_path, exist_ok=True)

                obs_jets["scalars"][:, -1] = np.ravel(pred_n)
                np.save(f'{save_path}/{obs_jets_path.split("/")[-1]}', obs_jets)
            else:
                save_path = f"{path}/figures"
                os.makedirs(save_path, exist_ok=True)
                
                # plot and compare to the truth
                pred_n_posterior=[]
                for nr, sample in tqdm(enumerate(dataloader), total=n_iters):
                    sample = {i:j.to(dev) for i,j in sample.items()}
                    pred_n_posterior.append(flow.sample(sample, 256).cpu().detach().numpy())
                    if nr==n_iters:
                        break

                pred_n_posterior = np.concatenate(pred_n_posterior, 0)[:, :, 0]
                truth_quantile = get_percentile(pred_n_posterior, truth_cnsts[:len(pred_n_posterior)], ["N"],numpy_version=True)
                
                # plt.figure()
                # plt.hist(pred_n_posterior[0], bins=30)
                
                n = len(pred_n)

                # single sample
                dist_styles = [{"label": "$N_{Top}$", "color": "black"},
                               {"label": "$N_{Flow}$", "color":"blue"}]
                fig, ax = plt.subplots(1,1, figsize=(8,6))
                plot.plot_hist(truth_cnsts, np.ravel(pred_n),
                               style={"bins":20, "range": [20, 175]},
                               log_yscale=False, ax=ax,
                               dist_styles = dist_styles)
                ax.set_xlabel("Number of constituents")
                if config.save_figures:
                    misc.save_fig(fig, f"{save_path}/{pileup_name}_single_sample.pdf")

                # truth quantiles sample
                dist_styles = [{"label": "Uniform", "color": "black"},
                               {"label": "Flow", "color":"blue"}]
                fig, ax = plt.subplots(1,1, figsize=(8,6))
                plot.plot_hist(np.random.uniform(0,100, (5_000_000,1)),
                               truth_quantile,
                               style={"bins":10}, ax=ax,
                               dist_styles =dist_styles)
                ax.set_xlabel("Truth quantiles")
                if config.save_figures:
                    misc.save_fig(fig, f"{save_path}/{pileup_name}_truth_quantiles.pdf")

                # single sample
                dist_styles = [{"label": r"($N_{Flow}$-$N_{Top}$)/$N_{Top}$", "color": "blue"}]
                fig, ax = plt.subplots(1,1, figsize=(8,6))
                plot.plot_hist((pred_n.flatten()-truth_cnsts[:n])/truth_cnsts[:n], style={"bins": np.linspace(-0.40,0.40, 20)},
                               ax=ax, dist_styles = dist_styles)
                ax.set_xlabel("Relative error of N")
                if config.save_figures:
                    misc.save_fig(fig, f"{save_path}/re_{pileup_name}_single_sample.pdf")
