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
from src.datamodule import MultiFileDataset, MultiStreamDataLoader, chunks
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
        self.max_pileup_size = kwargs.get("max_pileup_size", 400)
        self.noise_to_pileup = kwargs.get("noise_to_pileup", 0.0)
        self.n_pileup_to_select = kwargs.get("n_pileup_to_select", 0)
        self.max_cnstits = kwargs.get("max_cnstits", None)
        self.jet_norms = kwargs.get("jet_norms", None)
        self.mean = kwargs.get("mean", None)
        self.std = kwargs.get("std", None)
        
        self.pileup_cnts = kwargs.get("pileup_cnts", None)
        self.pileup_mask_cnts = kwargs.get("pileup_mask_cnts", None)
        

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
    
    def load_jet_cnts(self, paths):
        
        # load point clouts
        try:
            (cnts_vars, mask_cnts, _, _max_cnstits, n_pc, col_cnts
            ) = load_csv(paths, len(paths)>1, max_cnts=self.max_cnstits)
        except ValueError as e:
            print(e)
            raise ValueError(f"Path {self.jet_path}")
        # scale from MeV to GeV
        cnts_vars[...,np.in1d(col_cnts, self._MeV_to_GeV)
                  ] = cnts_vars[...,np.in1d(col_cnts, self._MeV_to_GeV)]/1000

        # which columns to defuse
        col_cnts_bool = np.isin(col_cnts, self.target_names)

        # calculate global jet variables
        jet_vars = self.physics_properties(cnts_vars[..., col_cnts_bool], mask_cnts)

        return cnts_vars, mask_cnts, _max_cnstits, n_pc, col_cnts, jet_vars


    def _load_data(self):
        # get jet data
        (self.cnts_vars, self.mask_cnts, self._max_cnstits, self.n_pc,
         self.col_cnts, self.jet_vars) = self.load_jet_cnts(self.jet_path)

        if self.max_cnstits is None:
            self.max_cnstits = self._max_cnstits
            self.max_pileup_size = self._max_cnstits
        
        self.col_cnts = np.array([f"cnts_{i}"for i in self.col_cnts])
        
        # which columns to defuse
        self.col_cnts_bool = np.isin(self.col_cnts,
                                    [f"cnts_{i}"for i in self.target_names])
        
        if self.jet_norms is None:
            self.jet_norms = {"mean":self.jet_vars.mean(0), "std":self.jet_vars.std(0)}

        # only to calculate mean/std
        cnts_vars_rel = self.relative_pos(
            self.cnts_vars[..., self.col_cnts_bool].copy(),self.jet_vars)

        self.mean["images"], self.std["images"] = self.calculate_norms(cnts_vars_rel,
                                                                    self.mask_cnts)
        
        # remove unused columns
        self.cnts_vars = self.cnts_vars[..., self.col_cnts_bool]
        
        # get jet data
        if (self.pileup_path is not None):
            if (self.pileup_cnts is None):
                self.pileup_cnts, self.pileup_mask_cnts, _,_, self.n_pileup_pc,_ =load_csv(self.pileup_path)
                # self.pileup_mean, self.pileup_std = self.calculate_norms(self.pileup_cnts[..., self.col_cnts_bool],
                #                                                 self.pileup_mask_cnts)
                
                # remove unused columns
                self.pileup_cnts = self.pileup_cnts[..., self.col_cnts_bool]
                
                # combine means/std TODO ask Johnny or chris how to do this?
                # Pileup should also contribute to mean/std right?

                # self.mean["cnts"] = (self.mean["cnts"]*self.mask_cnts.sum()+
                #                      pileup_mean*self.pileup_mask_cnts.sum())/(self.mask_cnts.sum()+self.pileup_mask_cnts.sum())
                # self.std["cnts"] = (self.std["cnts"]*self.mask_cnts.sum()+
                #                      pileup_std*self.pileup_mask_cnts.sum())/(self.mask_cnts.sum()+self.pileup_mask_cnts.sum())
            
            self.ctxt["pileup"] = {"cnts": self.pileup_cnts,
                                    "mask": self.pileup_mask_cnts}
    
    def get_pileup_data(self):
        return self.pileup_cnts, self.pileup_mask_cnts

    @staticmethod
    def calculate_norms(cnts_arr: np.ndarray, mask:np.ndarray):
        mean = cnts_arr[mask].mean(0)[None, :]
        std = cnts_arr[mask].std(0)[None, :]
        return mean, std
    
    @staticmethod
    def relative_pos(cnts_vars, jet_vars, reverse=False):
        "Calculate relative position to the jet"
        cnts_vars_rel = copy.deepcopy(cnts_vars)
        if reverse:
            for nr, i in enumerate(['eta', 'phi']):
                cnts_vars_rel[..., nr] += jet_vars[:, nr][:,None]
                if i in "phi":
                    cnts_vars_rel[..., nr][cnts_vars_rel[..., nr] >= np.pi] -= 2*np.pi
                    cnts_vars_rel[..., nr][cnts_vars_rel[..., nr] < -np.pi] += 2*np.pi

            cnts_vars_rel[..., nr+1] = (np.exp(cnts_vars_rel[..., nr+1])
                                        *jet_vars[..., nr+1][:,None])
        else:

            for nr, i in enumerate(['eta', 'phi']):
                cnts_vars_rel[..., nr] -= jet_vars[i].values[:,None]
                if i in "phi":
                    cnts_vars_rel[..., nr][cnts_vars_rel[..., nr] >= np.pi] -= 2*np.pi
                    cnts_vars_rel[..., nr][cnts_vars_rel[..., nr] < -np.pi] += 2*np.pi

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
                if "true_n_cnts" not in data:
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

        pileup_cnts = self.pileup_cnts[idx].copy()
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
    
    # @classmethod
    # def create_new_loader(cls, paths:list) -> None:
    def _load_new_data(self, paths:list) -> None:
        (self.cnts_vars, self.mask_cnts, self._max_cnstits, self.n_pc,
         self.col_cnts, self.jet_vars) = self.load_jet_cnts(paths)

        # relative cnts to the jet
        self.cnts_vars_rel = self.relative_pos(
            self.cnts_vars[..., self.col_cnts_bool].copy(),self.jet_vars
            )
        # data = JetPhysics([path], target_names=["eta", "phi", "pt"],
        #                     max_cnstits=cls.max_cnstits,
        #                     jet_scalars_cols = ["eta", "phi", "pt", "mass"],
        #                     jet_norms=cls.dataset.jet_norms,
        #                     std=cls.dataset.std, mean=cls.dataset.mean
        #                     )
        # return cls([paths], target_names=["eta", "phi", "pt"],
        #                     max_cnstits=cls.max_cnstits,
        #                     jet_scalars_cols = ["eta", "phi", "pt", "mass"],
        #                     jet_norms=cls.dataset.jet_norms,
        #                     std=cls.dataset.std, mean=cls.dataset.mean)

    def train_dataloader(self):
        return DataLoader(self, **self.loader_config)

    def test_dataloader(self): # TODO should not have train/test in the same
        test_loader_config = self.loader_config.copy()
        test_loader_config["persistent_workers"]=False
        test_loader_config["shuffle"]=False
        test_loader_config["pin_memory"]=False
        test_loader_config.pop("pin_memory_device", None)
        # test_loader_config["batch_size"]=512
        return DataLoader(self, **test_loader_config)

    def _shape(self):
        return [self.max_cnstits, len(self.target_names)]
    
    def __len__(self):
        return self.n_pc
    
    def __iter__(self):
        rand_lst = list(range(self.__len__()))
        np.random.shuffle(rand_lst)
        for idx in rand_lst:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        data = {}
        if ("pileup" in self.ctxt): # & (self.n_pileup_to_select>0):
            data["ctxt"] = {}
            
            # add pileup and recalculate jet vars
            if False:
                event, mask, jet_data = self.concat_pileup(idx,
                                                cnts_vars=self.cnts_vars[idx],
                                                mask_cnts_vars=self.mask_cnts[idx])
            else:
                jet_data = self.jet_vars.iloc[idx:idx+1]
                mask = self.mask_cnts[idx]
                event = self.relative_pos(self.cnts_vars[idx][None],jet_data)[0]

            # add data to dict
            data["images"] = self.relative_pos(self.cnts_vars[idx][None],jet_data)[0]
            data["images"] = np.float32((data["images"] -self.mean["images"])/self.std["images"])
            data["mask"] =  self.mask_cnts[idx]

            data["ctxt"]["cnts"] = np.float32((event-self.mean["images"])/self.std["images"])
            data["ctxt"]["mask"] = mask

            jet_var = (jet_data-self.jet_norms["mean"])/self.jet_norms["std"]
            jet_var = jet_var[self.jet_scalars_cols].values
            data["ctxt"]["scalars"] = np.float32(np.ravel(jet_var))
        else:
            cnts_vars_rel = self.relative_pos(
                self.cnts_vars[idx],self.jet_vars.iloc[idx:idx+1]
                ) # TODO i think shape is wrong
            data["images"] = np.float32(
                (cnts_vars_rel-self.mean["images"])/self.std["images"]
                )
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

        log = self.plot_marginals(self.cnts_vars[self.mask_cnts][:len(gen_cnts_flatten)],
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

def load_csv(paths: list, duplicate_number=False, max_cnts:int=None, verbose=False):
    "Load list of csv paths"
    df=[]
    usecols=["jetnumber ", " px ", " py ", " pz"]
    for i in tqdm(paths, disable=not verbose):
        if duplicate_number & (len(df)>0):
            _df = pd.read_csv(i, dtype=np.float32, engine="pyarrow")#, usecols=usecols)
            _df["jetnumber "] = _df["jetnumber "]+df[-1]["jetnumber "].iloc[-1]+1
            df.append(_df)
        else:
            df.append(pd.read_csv(i, dtype=np.float32,# engine="pyarrow",
                                #   usecols=usecols
                                  ))
    df = pd.concat(df, axis=0)
    df=df.rename(columns={i:i.replace(" ", "") for i in df.columns})
    event_nr_columns = [i for i in df.columns if "number" in i ]
    if all(np.in1d(["px", "py", "pz"],df.columns)):
        
        df["eta"],df["phi"],df["pt"]= detector_dimensions(df[["px", "py", "pz"]].values)
    cnts_df, mask = matrix_to_point_cloud(df.values, df[event_nr_columns].values, num_per_event_max=max_cnts)

    return cnts_df, mask, mask.sum(1).min(), mask.shape[-1], len(cnts_df), df.columns

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

class MultiJetFiles(MultiStreamDataLoader):
    
    def __init__(self, jet_path:str, target_names:list, sub_col:str, n_files:list=None,
                 jet_scalars_cols:list=None, max_cnstits:int=None, loader_config:dict=None,
                 **kwargs):
        self.jet_path=jet_path
        self.max_cnstits=max_cnstits
        self.sub_col=sub_col
        self.n_files=n_files
        self.target_names=target_names
        self.jet_scalars_cols=jet_scalars_cols
        self.loader_config=loader_config
        if self.loader_config is None:
            self.loader_config={}
        if self.jet_scalars_cols is None:
            self.jet_scalars_cols={}
        
        self.paths = misc.load_yaml(self.jet_path)
        self.num_workers=self.loader_config.pop("num_workers", 1)

        data_list = self.paths[f"{sub_col}_path"][:n_files]

        split_data_lst = list(chunks(data_list,
                                     int(np.ceil(len(data_list)/self.num_workers))))
        
        if self.num_workers < len(split_data_lst):
            raise ValueError("max_workers higher than number of fils")
        
        datasets = [MultiFileDataset(data_lst=path,
                                     batch_size=self.loader_config["batch_size"],
                                     processing_func=self._process_data)
                    for path in split_data_lst]
        self.jet_physics_cfg = self.__dict__.copy()
        self.jet_physics_cfg.pop("jet_path")
        
        self.jet_physics_cfg.update(kwargs)
        self.dataset = JetPhysics(jet_path = data_list[:kwargs.get("n_valid_files", 4)],
                                  **self.jet_physics_cfg)
        loader_config["num_workers"] = 8
        super().__init__(datasets=datasets,
                         data_kw=loader_config)


    def _shape(self): # TODO to be removed
        return self.dataset._shape()

    def train_dataloader(self): # TODO to be removed
        return self

    def test_dataloader(self): # TODO to be removed
        return self.train_dataloader()

    def get_ctxt_shape(self): # TODO to be removed
        return self.dataset.get_ctxt_shape()

        # return {
        #     #"cnts": [self.max_pileup_size, len(self.target_names)],
        #         "scalars": 4
        #         }

    def _process_data(self, paths_lst):
        for path in paths_lst:
            # self.dataset._load_new_data([path])
            data = JetPhysics(jet_path=[path], jet_norms=self.dataset.jet_norms,
                              std=self.dataset.std, mean=self.dataset.mean,
                              pileup_cnts = self.dataset.pileup_cnts[:10],
                              pileup_mask_cnts = self.dataset.pileup_mask_cnts[:10],
                              **self.jet_physics_cfg
                              )
            for i in data:
                yield i

if __name__ == "__main__":
    # %matplotlib widget
    if True:
        import matplotlib.pyplot as plt
        import hydra
        from tools import misc
        import sys

        PATH = ["/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"]
        PILEUP_PATH = ["/home/users/a/algren/scratch/diffusion/pileup/pileup.csv"]
        paths = misc.load_yaml("/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/path_lists.yaml")
        
        test_path = ["/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/topjets_518.csv"]
        physics = JetPhysics(test_path,
                             loader_config={"batch_size": 256,"num_workers": 4},
                             target_names=["eta", "phi", "pt"],
                             jet_scalars_cols=["eta", "phi", "pt", "mass"],
                             pileup_path=PILEUP_PATH,
                             n_pileup_to_select=100,
                             )
        # sys.exit()
        dataloader  = physics.train_dataloader()
        # data =[]
        # data_pileup =[]
        # for i in tqdm(dataloader):
        #     # data_pileup.append(i["ctxt"]["pc"][i["ctxt"]["mask"]])
        #     data.append(i["images"][i["mask"]])
        # data = T.concat(data, 0).numpy()

            
        
        
        # loader = MultiJetFiles()
        # loader = MultiStreamDataLoader(datasets, data_kw={"batch_size":1024})
        for ep in range(1):
            multi_data = {"images":[], "scalars":[], "images_ctxt":[], "mask": [],
                          "mask_ctxt": []}
            for i in tqdm(dataloader):
                multi_data["images"].append(i["images"])
                multi_data["mask"].append(i["mask"])
                
                multi_data["images_ctxt"].append(i["ctxt"]["cnts"])
                multi_data["mask_ctxt"].append(i["ctxt"]["mask"])
                
                multi_data["scalars"].append(i["ctxt"]["scalars"])
                
        for i in multi_data:
            multi_data[i] = T.concat(multi_data[i], 0).numpy()
        # Combine image channels for each variable
        bounds = [[-2.6, 2.6], [-2.6, 2.6]]
        bins = 30
        nr = 2
        for i,j in [["images", "mask"], ["images_ctxt", "mask_ctxt"]]:
            plt.figure(figsize=(8*1.5,6*1.5))
            cnts = multi_data[i][nr][multi_data[j][nr]]
            print(f"Tracks: {multi_data[j][nr].sum()}")
            plt.hist2d(
                x=cnts[..., 0],
                y=cnts[..., 1],
                weights=cnts[..., -1],
                bins=bins,
                range=bounds,
            )
            plt.xlabel(r"$\Delta \eta$", fontsize=15)
            plt.ylabel(r"$\Delta \phi$", fontsize=15)
        # plt.figure()
        # cnts = multi_data["images_ctxt"][nr][multi_data["mask_ctxt"][nr]]
        # plt.hist2d(
        #     x=cnts[..., 0],
        #     y=cnts[..., 1],
        #     weights=cnts[..., -1],
        #     bins=bins,
        #     range=bounds,
        # )
        
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
        
        if False:
            # data = physics.cnts_vars_rel[physics.mask_cnts]#
            for i in range(data.shape[1]):
            # for i in range(physics.cnts_vars.shape[-1]):
                counts, _ = plot.plot_hist(data[:,i],multi_data[:,i], style={"bins":250})
                # plt.hist(log_squash(physics.cnts_vars[physics.mask_cnts][:,i]), bins=50)
                plt.xlabel(physics.col_jets[3+i].replace("jet", "cnts"))
                print(np.mean(counts["dist_0"]["counts"][-1]/counts["dist_1"]["counts"][-1]))
                # plt.yscale("log")

        
    else:
        from sklearn.model_selection import train_test_split
        PATH = "/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/"
        if False:
            train_path, valid_path = train_test_split(glob(PATH+"*"), test_size=0.30)
            valid_path, test_path = train_test_split(valid_path, test_size=0.35)
            paths = {"train_path":train_path, "valid_path":valid_path, "test_path": test_path}
            misc.save_yaml(paths, f"{PATH}/path_lists.yaml")
    if False:
        
        from tools.visualization import general_plotting as plot
        if True:
            mass = physics_valid.jet_vars["mass"]
            mask_valid = physics_valid.jet_vars["pt"]>0
            mask_train = physics_train.jet_vars["pt"]>0
            # plt.hist(mass[mask], bins=50, range=[0, 300])
            for i in physics_train.jet_vars:
                plot.plot_hist(physics_train.jet_vars[mask_train][i],
                            physics_valid.jet_vars[mask_valid][i])
                plt.xlabel(i)
        else:
            train_data=[]
            for i in tqdm(physics_train.train_dataloader()):
                # train_data.append(i["ctxt"]["cnts"][i["ctxt"]["mask"]].numpy())
                # train_data.append(i["ctxt"]["scalars"].numpy())
                train_data.append(i["images"][i["mask"]].numpy())
            train_data = np.concatenate(train_data, 0)

            valid_data=[]
            for i in tqdm(physics_valid.train_dataloader()):
                # valid_data.append(i["ctxt"]["cnts"][i["ctxt"]["mask"]].numpy())
                # valid_data.append(i["ctxt"]["scalars"].numpy())
                valid_data.append(i["images"][i["mask"]].numpy())
            valid_data = np.concatenate(valid_data, 0)

        
        for i in range(physics_valid.cnts_vars.shape[-1]):
            plot.plot_hist(physics_valid.cnts_vars[physics_valid.mask_cnts][:,i],
                           physics_train.cnts_vars[physics_train.mask_cnts][:,i],
                           normalise=True)
        # for i in range(train_data.shape[-1]):
            # counts, ax = plot.plot_hist(train_data[:,i],
            #                valid_data[:,i],
            #                normalise=True)
            # print((np.array(counts["dist_0"]["counts"])/np.array(counts["dist_1"]["counts"])).mean())
            # plt.hist(log_squash(physics_valid.cnts_vars[physics_valid.mask_cnts][:,i]), bins=50)
            plt.xlabel("delta "+physics_valid.col_cnts[physics_valid.col_cnts_bool][i])
            # plt.xlabel(physics_valid.jet_scalars_cols[i])
            plt.yscale("log")