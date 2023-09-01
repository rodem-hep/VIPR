# from https://gitlab.cern.ch/mleigh/jetdiffusion/-/blob/matt_dev/src/physics.py
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from typing import Union

from tqdm import tqdm
from glob import glob
import os
import copy

import numpy as np
import torch as T
import pandas as pd
from torch.utils.data import Dataset,DataLoader


from tools import misc
from src.utils import undo_log_squash, log_squash
from src.eval_utils import EvaluateFramework
from src.prepare_data import fill_data_in_pc, matrix_to_point_cloud
from tools.transformations import log_squash, undo_log_squash

JET_COL = ["px","py","pz","eta", "phi", "pt", "mass"]

def calculate_pT(px, py):
    return np.sqrt(px**2+py**2)

def calculate_phi(px, py):
    return np.arctan2(py, px) # np.arccos(px/pT)

def calculate_eta(pz, pT):
    return np.arcsinh(pz/pT)

def detector_dimensions(df:np.ndarray):

    # calculate pT
    pT = calculate_pT(df[..., 0], df[..., 1])

    # calculate phi
    phi = calculate_phi(df[..., 0], df[..., 1])

    # calculate eta
    eta = calculate_eta(df[..., 2], pT)
    
    return eta[..., None], phi[..., None], pT[..., None]

def jet_variables(sample, mask):
    # calculate eta/phi/pT
    # eta, phi, pT = detector_dimensions(sample)

    # sample_epp = np.concatenate([eta, phi, pT], -1)
    
    # calculate summary jet features
    jet_vars = numpy_locals_to_mass_and_pt(sample, mask)
    jet_vars = pd.DataFrame(jet_vars, columns=JET_COL)
    
    
    return jet_vars

class JetPhysics(EvaluateFramework, Dataset):
    # TODO maybe dangerous that it is both a eval framework and dataloader
    def __init__(self, jet_path:Union[dict, list], target_names:list,
                 jet_scalars_cols:list=None, pileup_path:Union[str, list]=None, **kwargs):
        self.n_files = kwargs.get("n_files", None)
        if ".yaml" in jet_path:
            self.jet_path=misc.load_yaml(jet_path)[f"{kwargs['sub_col']}_path"][:self.n_files] # sub_col is required
        else:
            self.jet_path=jet_path

        self.standardize_bool = kwargs.get("standardize_bool", False)
        self.loader_config = kwargs.get("loader_config", {})
        self.max_pileup_size = kwargs.get("max_pileup_size", 384)
        self.noise_to_pileup = kwargs.get("noise_to_pileup", 0.0)
        self.n_pileup_to_select = kwargs.get("n_pileup_to_select", 0)
        self.pileup_path=pileup_path
        self.target_names = target_names
        self.col_jets = [f"jet_{i}"for i in JET_COL]
        self.hist_kwargs={"percentile_lst":[0, 99],
                     "style": {"bins": 40, "histtype":"step"},
                     "dist_styles":[
                        {"marker":"o", "color":"black", "label":"Truth", "linewidth":0},
                        {"linestyle": "dotted","color":"blue", "label":"Generated", "drawstyle":"steps-mid"}]
                     }
        self._MeV_to_GeV = ["px", "py", "pz", "pt"]
        self.mean, self.std = {}, {}
        self.ctxt = {}
        

        self.jet_scalars_cols = jet_scalars_cols

        if jet_path is not None:
            self._load_data()

    def _load_data(self):
        # get jet data
        (self.cnts_vars, self.mask_cnts, self.min_cnstits, self.max_cnstits, self.n_pc,
            self.col_cnts) = self.load_csv(self.jet_path, len(self.jet_path)>1)
        
        # MeV to GeV
        self.cnts_vars[...,np.in1d(self.col_cnts, self._MeV_to_GeV)] = self.cnts_vars[...,np.in1d(self.col_cnts, self._MeV_to_GeV)]/1000
        
        self.col_cnts = np.array([f"cnts_{i}"for i in self.col_cnts])
        
        # which columns to defuse
        self.col_cnts_bool = np.isin(self.col_cnts,
                                        [f"cnts_{i}"for i in self.target_names])
        
        self.jet_vars = self.physics_properties(self.cnts_vars[..., self.col_cnts_bool],
                                                self.mask_cnts)
        self.jet_norms = {"mean":self.jet_vars.mean(0), "std":self.jet_vars.std(0)}
        

        self.cnts_vars_rel = self.relative_pos(
            self.cnts_vars[..., self.col_cnts_bool].copy(),self.jet_vars)

        self.mean["cnts"], self.std["cnts"] = self.calculate_norms(self.cnts_vars_rel,
                                                                    self.mask_cnts)
        
        # get jet data
        if self.pileup_path is not None:
            (self.pileup_cnts, self.pileup_mask_cnts, self.min_pileup_cnstits,
                self.max_pileup_cnstits, self.n_pileup_pc, self.col_pileup_cnts
                )  = self.load_csv(self.pileup_path)
            self.pileup_mean, self.pileup_std = self.calculate_norms(self.pileup_cnts[..., self.col_cnts_bool],
                                                            self.pileup_mask_cnts)
            
            # combine means/std TODO ask Johnny or chris how to do this?
            # Pileup should also contribute to mean/std right?

            # self.mean["cnts"] = (self.mean["cnts"]*self.mask_cnts.sum()+
            #                      pileup_mean*self.pileup_mask_cnts.sum())/(self.mask_cnts.sum()+self.pileup_mask_cnts.sum())
            # self.std["cnts"] = (self.std["cnts"]*self.mask_cnts.sum()+
            #                      pileup_std*self.pileup_mask_cnts.sum())/(self.mask_cnts.sum()+self.pileup_mask_cnts.sum())
            
            self.ctxt["pileup"] = {"cnts": self.pileup_cnts[..., self.col_cnts_bool],
                                    "mask": self.pileup_mask_cnts}

    @staticmethod
    def calculate_norms(cnts_arr: np.ndarray, mask:np.ndarray):
        mean = cnts_arr[mask].mean(0)[None, :]
        std = cnts_arr[mask].std(0)[None, :]
        return mean, std

    def load_csv(self, paths: list, duplicate_number=False):
        "Load list of csv paths"
        df=[]
        for i in tqdm(paths):
            if duplicate_number & (len(df)>0):
                _df = pd.read_csv(i, dtype=np.float32, engine="pyarrow")
                _df["jetnumber "] = _df["jetnumber "]+df[-1]["jetnumber "].iloc[-1]+1
                df.append(_df)
            else:
                df.append(pd.read_csv(i, dtype=np.float32, engine="pyarrow"))
        df = pd.concat(df, axis=0)
        df=df.rename(columns={i:i.replace(" ", "") for i in df.columns})
        event_nr_columns = [i for i in df.columns if "number" in i ]
        if all(np.in1d(["px", "py", "pz"],df.columns)):
            
            df["eta"],df["phi"],df["pt"]= detector_dimensions(df[["px", "py", "pz"]].values)
        cnts_df, mask = matrix_to_point_cloud(df.values, df[event_nr_columns].values)

        return cnts_df, mask, mask.sum(1).min(), mask.sum(1).max(), len(cnts_df), df.columns
    
    @staticmethod
    def relative_pos(cnts_vars, jet_vars, reverse=False):
        "Calculate relative position to the jet"
        cnts_vars_rel = copy.deepcopy(cnts_vars)
        if reverse:
            for nr, i in enumerate(['eta', 'phi']):
                cnts_vars_rel[..., nr] += jet_vars[:, nr][:,None]

            cnts_vars_rel[..., nr+1] = (np.exp(cnts_vars_rel[..., nr+1])
                                        *jet_vars[..., nr+1][:,None])
        else:

            for nr, i in enumerate(['eta', 'phi']):
                cnts_vars_rel[..., nr] -= jet_vars[i].values[:,None]

            cnts_vars_rel[..., nr+1] = np.log(cnts_vars_rel[..., nr+1]/
                                              jet_vars["pt"].values[:,None])
            cnts_vars_rel[..., nr+1] = np.nan_to_num(cnts_vars_rel[..., nr+1], -1)

        return cnts_vars_rel

    def physics_properties(self, sample, mask):
        # calculate eta/phi/pT and jet variables
        return jet_variables(sample, mask)
    
    def get_normed_ctxt(self):
        # TODO construct pileup with ttbar events
        data={}
        test_dataloader = self.test_dataloader()
        for i in test_dataloader:
            if "mask" in i:
                if "mask" not in data:
                    data["true_n_cnts"] = i["mask"].sum(1).numpy()
                else:
                    data["true_n_cnts"] = np.concatenate([data["true_n_cnts"],
                                                          i["mask"].sum(1).numpy()])

            for i,j in i["ctxt"].items():
                if i not in data:
                    data[i] = j
                else:
                    data[i] = np.concatenate([data[i], j])
        return data
    
    def concat_pileup(self, idx_jet, cnts_vars, mask_cnts_vars):
        # generate events with pileup using in __getitem__

        jet_vars = self.jet_vars.iloc[idx_jet:idx_jet+1]
        
        idx = np.random.choice(np.arange(0, self.n_pileup_pc,1), self.n_pileup_to_select)

        pileup_cnts = self.pileup_cnts[idx][..., self.col_cnts_bool].copy()
        pileup_cnts_mask = self.pileup_mask_cnts[idx]

        pileup_cnts_rel = self.relative_pos(pileup_cnts,jet_vars)

        delta_R_mask = np.sqrt(np.sum(pileup_cnts_rel[..., :2]**2, -1))<1

        pileup_mask = delta_R_mask & pileup_cnts_mask
        
        # concat and fill full event matrix
        # TODO should pply be shuffled at some point
        # cnts_vars & pileup_cnts_rel should be shuffle individual and together
        evt_w_pileup = np.concatenate([cnts_vars[mask_cnts_vars],
                                       pileup_cnts[pileup_mask]],
                                      0)[:self.max_pileup_size]
        
        full_event = np.ones((self.max_pileup_size, cnts_vars.shape[-1]))*-999
        full_event[:len(evt_w_pileup)] = evt_w_pileup
        mask_events = np.any(full_event!=-999, -1)

        # full_event = full_event+T.randn_like(T.tensor(full_event)).numpy()*self.noise_to_pileup
        new_jet_vars = self.physics_properties(full_event, mask_events)

        # do not move jet
        new_jet_vars[["eta", "phi"]] = jet_vars[["eta", "phi"]].values

        # relative coords
        full_event = self.relative_pos(full_event[None],new_jet_vars)[0]
        
        return full_event, mask_events, new_jet_vars

    def get_ctxt_shape(self):
        return {"cnts": [self.max_pileup_size, len(self.target_names)],
                "scalars": len(self.jet_scalars_cols)}
    
    def get_diffusion_data(self):
        return (self.cnts_vars_rel, self.mask_cnts, self.jet_vars.values,
                self.min_cnstits, self.max_cnstits, self.n_pc)

    def train_dataloader(self):
        return DataLoader(self, **self.loader_config)

    def test_dataloader(self): # TODO should not have train/test in the same
        test_loader_config = self.loader_config.copy()
        test_loader_config["persistent_workers"]=False
        test_loader_config["shuffle"]=False
        return DataLoader(self, **test_loader_config)

    def _shape(self):
        return [self.max_cnstits, len(self.target_names)]
    
    def __len__(self):
        return self.n_pc

    def __getitem__(self, idx):
        data = {}
        if ("pileup" in self.ctxt): # & (self.n_pileup_to_select>0):
            data["ctxt"] = {}
            
            # add pileup and recalculate jet vars
            event, mask, jet_data = self.concat_pileup(idx,
                                             cnts_vars=self.cnts_vars[idx][:,self.col_cnts_bool],
                                             mask_cnts_vars=self.mask_cnts[idx])

            # add data to dict
            data["images"] = self.relative_pos(self.cnts_vars[idx][:,self.col_cnts_bool][None],jet_data)[0]
            data["images"] = np.float32((data["images"] -self.mean["cnts"])/self.std["cnts"])
            data["mask"] =  self.mask_cnts[idx]

            data["ctxt"]["cnts"] = np.float32((event-self.mean["cnts"])/self.std["cnts"])
            data["ctxt"]["mask"] = mask

            jet_var = (jet_data-self.jet_norms["mean"])/self.jet_norms["std"]
            jet_var = jet_var[self.jet_scalars_cols].values
            data["ctxt"]["scalars"] = np.float32(np.ravel(jet_var))
        else:
            data["images"] = np.float32(
                (self.cnts_vars_rel[idx]-self.mean["cnts"])/self.std["cnts"]),
            data["mask"] = self.mask_cnts[idx]
            if self.jet_scalars_cols is not None:
                data["ctxt"] = {}
                jet_var = self.jet_vars[self.jet_scalars_cols].iloc[idx:idx+1].values
                jet_var = ((jet_var-self.jet_norms["mean"][self.jet_scalars_cols].values)/
                        self.jet_norms["std"][self.jet_scalars_cols].values)
                data["ctxt"]["scalars"] = np.float32(np.ravel(jet_var))

        return data
    
    def __call__(self, gen_data, mask, ctxt):
        log={}
        if isinstance(gen_data, T.Tensor):
            gen_data = gen_data.cpu().detach().numpy()
        if isinstance(mask, T.Tensor):
            mask = mask.cpu().detach().numpy()
        # renorm
        if "scalars" in ctxt:
            ctxt_scalars = (ctxt["scalars"]
                            *self.jet_norms["std"][self.jet_scalars_cols].values
                            +self.jet_norms["mean"][self.jet_scalars_cols].values)

            if isinstance(ctxt_scalars, T.Tensor):
                ctxt_scalars = ctxt_scalars.cpu().detach().numpy()

            # reverse relative
            gen_cnts_vars = self.relative_pos(gen_data, ctxt_scalars, reverse=True)
            generated_jet_vars = self.physics_properties(gen_cnts_vars, mask==1)

            log = self.plot_marginals(self.jet_vars[self.jet_scalars_cols].values,
                                    generated_jet_vars[self.jet_scalars_cols].values,
                                    col_name=[f"jet_{i}" for i in self.jet_scalars_cols],
                                    hist_kwargs=self.hist_kwargs, log=log)

        gen_cnts_flatten = gen_cnts_vars[mask==1]

        log = self.plot_marginals(self.cnts_vars[self.mask_cnts][:len(gen_cnts_flatten),
                                                                 self.col_cnts_bool],
                                  gen_cnts_flatten,
                                  col_name=self.col_cnts[self.col_cnts_bool], log=log,
                                  hist_kwargs=self.hist_kwargs)

        return log
    
class JetLoader(JetPhysics):
    def __init__(self, path:str, sub_col:str, target_names:list,
                 jet_scalars_cols:list=None, **kwargs):
        paths = misc.load_yaml(path)[sub_col+"_path"]
        super().__init__(None, target_names=target_names, jet_scalars_cols=jet_scalars_cols,
                         **kwargs)
        self.paths=paths
        self.train_data = self.load_data(paths[:10])
           
    def load_data(self, path):
        # get jet data
        self._load_data(path)

def torch_locals_to_mass_and_pt(
    csts: T.Tensor,
    mask: T.BoolTensor,
    undo_logsquash: bool = False,
) -> T.Tensor:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - eta
    - phi
    - pt or log_squash_pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = undo_log_squash(csts[..., 2]) if undo_logsquash else csts[..., 2]

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    jet_px = (pt * T.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * T.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * T.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * T.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = T.clamp_min(jet_px**2 + jet_py**2, 0).sqrt()
    jet_m = T.clamp_min(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0).sqrt()

    return T.vstack([jet_pt, jet_m]).T


def numpy_locals_to_mass_and_pt(csts: np.ndarray, mask: np.ndarray,
                                undo_logsquash: bool = False
                                ) -> np.ndarray:
    """Calculate the overall jet pt and mass from the constituents. The
    constituents are expected to be expressed as:

    - eta
    - phi
    - pt
    """

    # Calculate the constituent pt, eta and phi
    eta = csts[..., 0]
    phi = csts[..., 1]
    pt = undo_log_squash(csts[..., 2]) if undo_logsquash else csts[..., 2]
    

    # Calculate the total jet values in cartensian coordinates, include mask for sum
    if len(mask.shape)==1:
        jet_px = (pt[mask] * np.cos(phi[mask])).sum(axis=-1)
        jet_py = (pt[mask] * np.sin(phi[mask])).sum(axis=-1)
        jet_pz = (pt[mask] * np.sinh(eta[mask])).sum(axis=-1)
        jet_e = (pt[mask] * np.cosh(eta[mask])).sum(axis=-1)
    else:
        jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
        jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
        jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
        jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)
    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))
    
    # position of jet
    jet_phi = calculate_phi(jet_px, jet_py)
    jet_eta = calculate_eta(jet_pz, jet_pt)

    return np.vstack([jet_px, jet_py, jet_pz, jet_eta, jet_phi, jet_pt, jet_m]).T

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    if False:
        PATH = ["/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"]
        PILEUP_PATH = ["/home/users/a/algren/scratch/diffusion/pileup/pileup.csv"]
        
        physics = JetPhysics(PATH, pileup_path=PILEUP_PATH, loader_config={"batch_size": 256,
                                                                        "num_workers": 4})
        
        dataloader  = physics.train_dataloader()
        data =[]
        data_pileup =[]
        for i in tqdm(dataloader):
            data_pileup.append(i["ctxt"]["pc"][i["ctxt"]["mask"]])
            data.append(i["images"][i["mask"]])
        
        # # generate events with pileup
        # idx_jet = 0
        # events_with_pileup, mask = physics.concat_pileup(0)
        
        # for i in range(3):
        #     plt.figure()
        #     _,bins, _ =plt.hist(events_with_pileup[mask, i+3], bins=50, density=True, label="jet cnts+pileup cnts", alpha=0.5)
        #     _, bins, _ = plt.hist(physics.cnts_vars[idx_jet][physics.mask_cnts[idx_jet]][:,i+3], bins=bins, density=True, label="jet cnts", alpha=0.5)
        #     # plt.hist(pileup_cnts_rel[pileup_cnts_mask & delta_R_mask][:,i], bins=bins, density=True, label="pileup cnts", alpha=0.5)
        #     # plt.xlabel(r"$\Delta$"+physics.col_cnts[i+3])
        #     plt.xlabel(physics.col_cnts[i+3])
        #     # plt.legend(title=f"Total cnts {np.sum(physics.mask_cnts[idx_jet])+np.sum(pileup_cnts_mask & delta_R_mask)}")
        #     plt.legend(title=f"Total cnts")
        
        if True:
            # data = physics.cnts_vars_rel[physics.mask_cnts]#
            data = T.concat(data, 0).numpy()
            for i in range(data.shape[1]):
            # for i in range(physics.cnts_vars.shape[-1]):
                plt.figure(figsize=(4,3))
                plt.hist(data[:,i], bins=50)
                # plt.hist(log_squash(physics.cnts_vars[physics.mask_cnts][:,i]), bins=50)
                plt.xlabel(physics.col_jets[3+i].replace("jet", "cnts"))
                # plt.yscale("log")

        
    else:
        from sklearn.model_selection import train_test_split
        PATH = "/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/"
        if False:
            train_path, valid_path = train_test_split(glob(PATH+"*"), test_size=0.30)
            valid_path, test_path = train_test_split(valid_path, test_size=0.35)
            paths = {"train_path":train_path, "valid_path":valid_path, "test_path": test_path}
            misc.save_yaml(paths, f"{PATH}/path_lists.yaml")
        else:
            import hydra
            cfg_data = misc.load_yaml("/home/users/a/algren/work/diffusion/configs/data_cfgs/pc_top_jet.yaml")
            physics_train = hydra.utils.instantiate(cfg_data.train, n_files=20)
            physics_valid = hydra.utils.instantiate(cfg_data.valid, n_files=2)
            # next(iter(physics_train))
    if True:
        from tools.visualization import general_plotting as plot
        mass = physics_valid.jet_vars["mass"]
        mask_valid = physics_valid.jet_vars["pt"]>0
        mask_train = physics_train.jet_vars["pt"]>0
        # plt.hist(mass[mask], bins=50, range=[0, 300])
        for i in physics_train.jet_vars:
            plot.plot_hist(physics_train.jet_vars[mask_train][i],
                           physics_valid.jet_vars[mask_valid][i])
            plt.xlabel(i)
        for i in range(physics_valid.cnts_vars.shape[-1]):
        # for i in range(physics_valid.cnts_vars.shape[-1]):
            plot.plot_hist(physics_valid.cnts_vars[physics_valid.mask_cnts][:,i],
                           physics_train.cnts_vars[physics_train.mask_cnts][:,i],
                           normalise=True)
            # plt.hist(log_squash(physics_valid.cnts_vars[physics_valid.mask_cnts][:,i]), bins=50)
            plt.xlabel(physics_valid.col_cnts[i])
            plt.yscale("log")
            
            
            
            
            
            
            
            