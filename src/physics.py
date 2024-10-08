import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from typing import Union, List

from tqdm import tqdm
from glob import glob
import os
import copy
from scipy.stats import norm

import numpy as np
import torch as T
import pandas as pd
from torch.utils.data import Dataset,DataLoader


from tools import misc
import tools.visualization.general_plotting as plot
from src.utils import undo_log_squash, log_squash
from src.eval_utils import EvaluateFramework
from tools.datamodule.prepare_data import fill_data_in_pc, matrix_to_point_cloud
from tools.datamodule.pipeline import pc_2_image 
from tools.datamodule.datamodule import MultiFileDataset, MultiStreamDataLoader, chunks
from tools.transformations import log_squash, undo_log_squash

def rescale_phi(phi):
    phi[phi >= np.pi] -= 2*np.pi
    phi[phi < -np.pi] += 2*np.pi
    return phi
    
    
def relative_pos(cnts_vars, jet_vars, mask, reverse=False):
    "Calculate relative position to the jet"
    cnts_vars_rel = copy.deepcopy(cnts_vars)
    if reverse:
        # from relative position to abs position
        for nr, i in enumerate(['eta', 'phi']):
            cnts_vars_rel[..., nr] += jet_vars[:, nr][:,None]
            if i in "phi":
                cnts_vars_rel[..., nr] = rescale_phi(cnts_vars_rel[..., nr])

        cnts_vars_rel[..., nr+1] = np.exp(cnts_vars_rel[..., nr+1])
        # cnts_vars_rel[..., nr+1] = (np.exp(cnts_vars_rel[..., nr+1])*
                                        #   jet_vars[:, nr+1][:,None])
    else:
        # from abs position to relative position
        for nr, i in enumerate(['eta', 'phi']):
            cnts_vars_rel[..., nr] -= jet_vars[i].values[:,None]
            if i in "phi":
                cnts_vars_rel[..., nr] = rescale_phi(cnts_vars_rel[..., nr])

        # log squash pT
        cnts_vars_rel[..., nr+1][mask] = np.clip(cnts_vars_rel[..., nr+1][mask],
                                                    a_min=1e-8, a_max=None)

        cnts_vars_rel[..., nr+1][mask] = np.log(cnts_vars_rel[..., nr+1][mask])
        
        cnts_vars_rel[..., nr+1] = np.nan_to_num(cnts_vars_rel[..., nr+1], -1)


    if mask is not None:
        cnts_vars_rel[~mask]=0
    
    return cnts_vars_rel

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

class PileupDist:
    def __init__(self, mu, std):
        self.mu=mu
        self.std=std

    def __call__(self, size):
        return np.abs(np.int64(np.random.normal(self.mu, self.std, size)))

class JetPhysics(EvaluateFramework, Dataset):
    def __init__(self, jet_path: Union[dict, list], target_names: List[str],
                 jet_scalars_cols: List[str] = None,
                 pileup_path: Union[str, list] = None, **kwargs):
        super().__init__()  # Initialize parent classes if needed

        # Initialize instance variables
        self.jet_scalars_cols = jet_scalars_cols
        self.pileup_path = pileup_path
        self.target_names = target_names
        self.jet_path = jet_path

        # Set default values for optional arguments using kwargs.get()
        self.n_files = kwargs.get("n_files")
        self.loader_config = kwargs.get("loader_config", {})
        self.max_ctxt_cnstits = kwargs.get("max_ctxt_cnstits", 400)
        self.max_cnstits = kwargs.get("max_cnstits")
        
        # change the eta phi of the obs jet in conds
        self.move_pileup_jet = kwargs.get("move_pileup_jet", True)

        # init pileup class
        self.noise_to_pileup = kwargs.get("noise_to_pileup", 0.0)
        self.pileup_dist_args = kwargs.get("pileup_dist_args",{"mu": 0, "std": 0})
        self.pileup_dist = PileupDist(**self.pileup_dist_args)

        self.jet_norms = kwargs.get("jet_norms")
        self.datatype = kwargs.get("datatype", "pc")
        self.pileup_cnts = kwargs.get("pileup_cnts")
        self.pileup_mask_cnts = kwargs.get("pileup_mask_cnts")
        self._MeV_to_GeV = kwargs.get("MeV_2_GeV", ["px", "py", "pz", "pt"])
        self.image_style_kwargs = kwargs.get("image_style_kwargs",
                                       {"range":[[-2.6, 2.6], [-2.6, 2.6]],
                                        "bins":64})

        # Initialize other instance variables
        self.col_jets = [f"jet_{i}" for i in JET_COL]
        self.hist_kwargs = {
            "percentile_lst": [0, 99.5],
            "style": {"bins": 40, "histtype": "step"},
            "dist_styles": [
                {"marker": "o", "color": "black", "label": "Target", "linewidth": 0},
                {"linestyle": "dotted", "color": "blue", "label": "Diffusion", "drawstyle": "steps-mid"},
            ]
        }
        self.hist_kwargs_diff = copy.deepcopy(self.hist_kwargs)
        self.hist_kwargs_diff["dist_styles"] =  [
                {"linestyle": "dotted", "color": "blue", "label": "Target-Diffusion",
                 "drawstyle": "steps-mid"}]
        self.hist_kwargs_diff["percentile_lst"] = [1, 99]

        # Load jet path if it's a YAML file
        if ".yaml" in self.jet_path:
            self.jet_path = misc.load_yaml(self.jet_path)[f"{kwargs['sub_col']}_path"][:self.n_files]
        else:
            self.jet_path = jet_path

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
        
        jet_vars["n_cnts"] = mask_cnts.sum(1)

        return cnts_vars, mask_cnts, _max_cnstits, n_pc, col_cnts, jet_vars


    def _load_data(self):
        # get jet data
        (self.cnts_vars, self.mask_cnts, self._max_cnstits, self.n_pc,
         self.col_cnts, self.jet_vars) = self.load_jet_cnts(self.jet_path)

        if self.max_cnstits is None:
            self.max_cnstits = self._max_cnstits
            
        self.col_cnts = np.array([f"cnts_{i}"for i in self.col_cnts])
        
        # which columns to defuse
        self.col_cnts_bool = np.isin(self.col_cnts,
                                    [f"cnts_{i}"for i in self.target_names])
        
        if self.jet_norms is None:
            self.jet_norms = {"mean":self.jet_vars.mean(0), "std":self.jet_vars.std(0)}

        # only to calculate mean/std
        self.cnts_vars_rel = self.relative_pos(
            self.cnts_vars[..., self.col_cnts_bool].copy(),self.jet_vars,
            mask=self.mask_cnts)
        
        # print(f"ratio: {mask_dR.mean()}")
        
        # apply the dR cut
        # (self.jet_vars, self.cnts_vars, self.cnts_vars_rel, self.mask_cnts,
        #  self.n_pc) = (
        #     self.jet_vars[mask_dR],
        #     self.cnts_vars[mask_dR],
        #     self.cnts_vars_rel[mask_dR],
        #     self.mask_cnts[mask_dR],
        #     mask_dR.sum()
        #     )
     

        # self.mean["images"], self.std["images"] = self.calculate_norms(self.cnts_vars_rel,
        #                                                             self.mask_cnts)
        
        # remove unused columns
        self.cnts_vars = self.cnts_vars[..., self.col_cnts_bool]
        
        # get jet data
        if (self.pileup_path is not None):
            if (self.pileup_cnts is None):
                
                # import pileup events
                self.pileup_cnts, self.pileup_mask_cnts, _,_, self.n_pileup_pc,_ = load_csv(self.pileup_path)
                
                # remove unused columns
                self.pileup_cnts = self.pileup_cnts[..., self.col_cnts_bool]

            else:
                self.n_pileup_pc = len(self.pileup_cnts)
    
    def get_pileup_data(self):
        return self.pileup_cnts, self.pileup_mask_cnts
    
    @staticmethod
    def relative_pos(cnts_vars, jet_vars, mask, reverse=False):
        return relative_pos(cnts_vars=cnts_vars, jet_vars=jet_vars, mask=mask,
                            reverse=reverse)

    def physics_properties(self, sample, mask):
        # calculate eta/phi/pT and jet variables
        return jet_variables(sample, mask)
    
    def get_normed_ctxt(self, return_truth:bool=False):
        # TODO construct pileup with ttbar events
        data={}
        test_dataloader = self.test_dataloader()
        for i in test_dataloader:
            
            # unpack size
            if "mask" in i:
                if "true_n_cnts" not in data:
                    data["true_n_cnts"] = i["mask"].sum(1).numpy()
                else:
                    data["true_n_cnts"] = np.concatenate([data["true_n_cnts"],
                                                          i["mask"].sum(1).numpy()])
            # # get target for that context
            # if return_truth:
            #     for col in ["images", "mask"]:
            #         if col not in data_true:
            #             data_true[col] = i[col]
            #         else:
            #             data_true[col] = np.concatenate([data_true[col], i[col]])
            # get context
            for i,j in i["ctxt"].items():
                if i not in data:
                    data[i] = j
                else:
                    data[i] = np.concatenate([data[i], j])


        if return_truth:
            data_true = {"images": self.cnts_vars_rel, "mask": self.mask_cnts,
                         "true_n_cnts": self.jet_vars["n_cnts"].values,
                         "scalars": self.jet_vars[self.jet_scalars_cols[:4]].values}

            return data, data_true
        else:
            return data
    
    def concat_pileup(self, idx_jet, cnts_vars, mask_cnts_vars):
        # generate events with pileup using in __getitem__

        jet_vars = self.jet_vars.iloc[idx_jet:idx_jet+1]
        
        # generate mu
        mu = self.pileup_dist(1)
        
        # selecting mu number of pileup events
        idx = np.random.choice(self.n_pileup_pc, mu)

        pileup_cnts = self.pileup_cnts[idx].copy()[self.pileup_mask_cnts[idx]]
        
        # flip in eta        
        pileup_cnts[..., 0] = pileup_cnts[..., 0]*np.random.choice([-1, 1], len(pileup_cnts))
        
        # rotate in phi
        pileup_cnts[..., 1] += ((2*np.random.rand(len(pileup_cnts))-1)*np.pi)
        pileup_cnts[..., 1][pileup_cnts[..., 1] >= np.pi] -= 2*np.pi
        pileup_cnts[..., 1][pileup_cnts[..., 1] < -np.pi] += 2*np.pi

        # get relative position
        pileup_cnts_rel = self.relative_pos(pileup_cnts[None],jet_vars,
                                            mask = np.ones((1, len(pileup_cnts)))==1)

        # dR cut on the pileup cnts
        delta_R_mask = np.sqrt(np.sum(pileup_cnts_rel[0][..., :2]**2,-1))<=1
        
        
        # concat and fill full event matrix
        # TODO should pply be shuffled at some point
        # cnts_vars & pileup_cnts_rel should be shuffle individual and together
        evt_w_pileup = np.concatenate([cnts_vars[mask_cnts_vars],
                                       pileup_cnts[delta_R_mask]],
                                      0)[:self.max_ctxt_cnstits]
        
        full_event = np.ones((self.max_ctxt_cnstits, cnts_vars.shape[-1]))*-999
        full_event[:len(evt_w_pileup)] = evt_w_pileup
        mask_events = np.any(full_event!=-999, -1)

        # full_event = full_event+T.randn_like(T.tensor(full_event)).numpy()*self.noise_to_pileup
        new_jet_vars = self.physics_properties(full_event, mask_events)

        # do not move jet
        if not self.move_pileup_jet:
            new_jet_vars[["eta", "phi"]] = jet_vars[["eta", "phi"]].values
            
        # add true jet size
        new_jet_vars["n_cnts"] = jet_vars["n_cnts"].values

        # input mu into jet vars
        new_jet_vars["mu"] = mu
        
        # relative coords
        full_event = self.relative_pos(full_event[None],new_jet_vars,
                                       mask=mask_events[None])
        
        return full_event[0], mask_events, new_jet_vars

    def get_ctxt_shape(self):
        return {"cnts": [self.max_ctxt_cnstits, len(self.target_names)],
                "scalars": len(self.jet_scalars_cols)}

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
    
    def pc_2_image(self, data):
        return pc_2_image(data, style_kwargs=self.image_style_kwargs)

    def _shape(self):
        return {"images": [self.max_cnstits,
                           len(self.target_names)],
                "ctxt_images": [self.max_ctxt_cnstits, 
                                len(self.target_names)],
                "ctxt_scalars": [len(self.jet_scalars_cols)]}
    
    def __len__(self):
        return self.n_pc
    
    def __iter__(self):
        rand_lst = list(range(self.__len__()))
        np.random.shuffle(rand_lst)
        for idx in rand_lst:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        jet_var=None
        data = {"images": None}
        if self.pileup_cnts is not None: # & (self.n_pileup_to_select>0):
            data["ctxt"] = {}
            
            # add pileup and recalculate jet vars
            if True:
                event, mask, jet_var = self.concat_pileup(idx,
                                                cnts_vars=self.cnts_vars[idx],
                                                mask_cnts_vars=self.mask_cnts[idx])
            else:
                jet_var = self.jet_vars.iloc[idx:idx+1]
                mask = self.mask_cnts[idx]
                event = self.relative_pos(self.cnts_vars[idx][None],jet_var)
                event[~mask]=0

            # add data to dict
            if "pc" in self.datatype:
                data["mask"] =  self.mask_cnts[idx]
                # data["images"] = self.cnts_vars_rel[idx]
                data["images"] = self.relative_pos(self.cnts_vars[idx][None],jet_var, mask=data["mask"][None])[0]
                data["images"][~data["mask"]]=0

                data["images"] = np.float32(data["images"])

            elif self.datatype == "N":
                data["images"] = np.array([np.float32(self.mask_cnts[idx].sum())])

            # insert cnts into ctxt
            data["ctxt"]["cnts"] = np.float32(event)
            data["ctxt"]["mask"] = mask
            data["ctxt"]["cnts"][~data["ctxt"]["mask"]]=0

        else:
            cnts_vars_rel = self.relative_pos(
                self.cnts_vars[idx][None],self.jet_vars.iloc[idx:idx+1],
                mask=self.mask_cnts[idx]
                ) # TODO i think shape is wrong
            data["images"] = np.float32(cnts_vars_rel)
            data["mask"] = self.mask_cnts[idx]

        # Get scalar ctxt vars
        if self.jet_scalars_cols is not None:
            # Ensure that ctxt is present
            if "ctxt" not in data:
                data["ctxt"] = {}

            # check if jet_var is missing
            if jet_var is None:
                jet_var = self.jet_vars.iloc[idx:idx+1]

            # Fill in Jet scalars
            data["ctxt"]["scalars"] = np.float32(
                np.ravel(jet_var[self.jet_scalars_cols])
                )
        
        # convert to image
        if "image" in self.datatype:
            data["images"] = self.pc_2_image(data["images"])[None]
            data.pop("mask")
            if "cnts" in data["ctxt"]:
                data["ctxt"]["cnts"] = self.pc_2_image(data["ctxt"]["cnts"])[None]
                data["ctxt"].pop("mask")

        return data
    
    def __call__(self, gen_data, mask, ctxt, **kwargs):
        log=kwargs.get("log", {})
        if isinstance(gen_data, T.Tensor):
            gen_data = gen_data.cpu().detach().numpy()
        if isinstance(mask, T.Tensor):
            mask = mask.cpu().detach().numpy()

        # renorm
        if "scalars" in ctxt:
            ctxt_scalars = ctxt["scalars"]
            if isinstance(ctxt_scalars, T.Tensor):
                ctxt_scalars = ctxt_scalars.cpu().detach().numpy()
            
            # Ensure that columns in self.jet_vars
            scalars_cols = np.array(self.jet_scalars_cols)[np.in1d(self.jet_scalars_cols, self.jet_vars.keys())]
            
            # also remove n_cnts
            scalars_cols = scalars_cols[~np.in1d(scalars_cols, ["n_cnts"])]
            
            # get true jet HLV
            true_jet_hlv = self.jet_vars[scalars_cols].values[:len(ctxt_scalars)]

            # Plot raw ctxt
            log = self.plot_marginals(true_jet_hlv,ctxt_scalars,
                col_name=[f"jet_ctxt_{i}" for i in scalars_cols],
                hist_kwargs=self.hist_kwargs, log=log
                )

            # reverse relative
            gen_cnts_vars = self.relative_pos(gen_data, ctxt_scalars, mask=mask==1, reverse=True)
            
            # calculate jet var
            generated_jet_vars = self.physics_properties(gen_cnts_vars, mask==1)

            # Plot difference between jet var & generated jet var
            log = self.plot_marginals(
                true_jet_hlv, generated_jet_vars[scalars_cols].values,
                col_name=[f"jet_generated_{i}" for i in scalars_cols],
                hist_kwargs=self.hist_kwargs, log=log)

            response_gen = ((generated_jet_vars[scalars_cols].values -true_jet_hlv)
                            /true_jet_hlv)
            
            # Plot difference between jet var & generated jet var
            log = self.plot_marginals(np.nan_to_num(response_gen,-999,-999),
                col_name=[f"jet_generated_diff_{i}" for i in scalars_cols],
                ratio_bool=False, hist_kwargs=self.hist_kwargs_diff, log=log)

        gen_cnts_flatten = gen_cnts_vars[mask==1]

        # Plot marginals of cnts
        log = self.plot_marginals(self.cnts_vars[self.mask_cnts][:len(gen_cnts_flatten)],
                                  gen_cnts_flatten,
                                  col_name=self.col_cnts[self.col_cnts_bool], log=log,
                                  hist_kwargs=self.hist_kwargs)

        # Plot centralised marginals of cnts
        log = self.plot_marginals(self.cnts_vars_rel[self.mask_cnts][:len(gen_cnts_flatten)],
                                  gen_data[mask==1],
                                  col_name=[f"{i}_rel" for i in self.col_cnts[self.col_cnts_bool]], log=log,
                                  hist_kwargs=self.hist_kwargs)

        # proper styling for the images
        style={"range":[[-2.5, 2.5], [-np.pi, np.pi]], "bins":64}

        # plot pc as image
        gen_images = np.clip(np.log(pc_2_image(gen_cnts_vars[:100], mask=mask[:100]==1, style_kwargs=style)+1), 0, 1)

        # Upload images
        log.update(self.plot_images(gen_images[..., None], name="gen_image"))
        
        # create & upload ctxt & truth
        if kwargs.get("n_epoch", 0)==0:
            ctxt_cnts = self.relative_pos(ctxt["cnts"], ctxt_scalars, mask=ctxt["mask"]==1,
                                          reverse=True).numpy()
            truth_images = np.clip(np.log(pc_2_image(self.cnts_vars[:100], mask=self.mask_cnts[:100], style_kwargs=style)+1), 0,1)
            ctxt_images = np.clip(np.log(pc_2_image(ctxt_cnts[:100], mask=ctxt["mask"][:100]==1, style_kwargs=style)+1), 0, 1)
            log.update(self.plot_images(truth_images[..., None], name="truth_image"))
            log.update(self.plot_images(ctxt_images[..., None], name="ctxt_image"))

        return log

def load_csv(paths: list, duplicate_number=False, max_cnts:int=None, verbose=False):
    "Load list of csv paths"
    df=[]
    for i in tqdm(paths, disable=not verbose):
        _df = pd.read_csv(i, dtype=np.float32, engine="c")
        # _df = pd.read_csv(i, dtype=np.float32, engine="pyarrow")
        _df.rename(columns=lambda x: x.strip(), inplace=True)
        if duplicate_number & (len(df)>0):
            _df["jetnumber"] = _df["jetnumber"]+df[-1]["jetnumber"].iloc[-1]+1
        df.append(_df)

    df = pd.concat(df, axis=0)

    event_nr_columns = [i for i in df.columns if "number" in i ]
    if all(np.in1d(["px", "py", "pz"],df.columns)):
        # calculate eta/phi/pT
        df["eta"],df["phi"],df["pt"]= detector_dimensions(df[["px", "py", "pz"]].values)
    
    #pT cut
    if df["pt"].max()>1000:
        df = df[df["pt"]>1000] # pT in MeV
    else:
        df = df[df["pt"]>1] # pT in GeV
    
    # create PC
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
    def __init__(self, jet_path:str, sub_col:str, jet_physics_cfg:dict,
                 n_files:list=None,  loader_config:dict=None, **kwargs):
        self.jet_path=jet_path
        self.sub_col=sub_col
        self.n_files=n_files
        self.loader_config=loader_config
        self.jet_physics_cfg = jet_physics_cfg
        if self.loader_config is None:
            self.loader_config={}

        self.num_workers=self.loader_config.pop("num_workers", 1)

        self.paths = misc.load_yaml(self.jet_path)

        data_list = self.paths[f"{sub_col}_path"][:n_files]

        split_data_lst = list(chunks(data_list,
                                     int(np.ceil(len(data_list)/self.num_workers))))
        
        if self.num_workers < len(split_data_lst):
            raise ValueError("max_workers higher than number of files")
        
        datasets = [MultiFileDataset(data_lst=path,
                                     batch_size=self.loader_config["batch_size"],
                                     processing_func=self._process_data)
                    for path in split_data_lst]
        # datasets = [len(JetPhysics(jet_path=path,**self.jet_physics_cfg))
        #             for path in split_data_lst]
        
        self.dataset = JetPhysics(jet_path = data_list[:kwargs.get("n_valid_files", 4)],
                                  **self.jet_physics_cfg)
        
        # calculate number of batches. There are 9999 events in each file
        self.n_batches = sum([len(i) for i in split_data_lst])*9999//loader_config["batch_size"]
        loader_config["num_workers"] = 1

        super().__init__(datasets=datasets, data_kw=loader_config)
    
    def __len__(self):
        return self.n_batches

    def _shape(self): # TODO to be removed
        return self.dataset._shape()

    def train_dataloader(self): # TODO to be removed
        return self

    def test_dataloader(self): # TODO to be removed
        return self.train_dataloader()

    def get_ctxt_shape(self): # TODO to be removed
        return self.dataset.get_ctxt_shape()

    def _process_data(self, paths_lst):
        for path in paths_lst:
            # self.dataset._load_new_data([path])
            data = JetPhysics(jet_path=[path], jet_norms=self.dataset.jet_norms,
                              pileup_cnts = self.dataset.pileup_cnts,
                              pileup_mask_cnts = self.dataset.pileup_mask_cnts,
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
        save_fig=False
        PATH = ["/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"]
        PILEUP_PATH = ["/home/users/a/algren/scratch/diffusion/pileup/pileup.csv"]
        paths = misc.load_yaml("/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/path_lists.yaml")
        
        # paths = glob('/srv/beegfs/scratch/groups/rodem/datasets/RODEMJetTagging/train/ttbar/*')
        # import h5py
        # data = h5py.File(paths[0], 'r') as h5_file
        
        # sys.exit()
        test_path = paths["test_path"]
        
        if False: # count total number of cnts
            test_path = glob("/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/*.csv")
            max_cns=[]
            total_cns=[]
            size=0
            nested_lists = [test_path[i:i+10] for i in range(0, len(test_path), 10)]
            for i in tqdm(nested_lists):
                physics = JetPhysics(i,
                                    loader_config={"batch_size": 256,"num_workers": 4},
                                    target_names=["eta", "phi", "pt"],
                                    jet_scalars_cols=["eta", "phi", "pt", "mass", "mu"],
                                    device="cpu",
                                    # pileup_path=PILEUP_PATH,
                                    # datatype="image",
                                    # pileup_dist_args={"mu":15, "std":5}
                                    )
                # data = load_csv(i, len(paths)>1)
                size+=len(physics)
                max_cns.append(physics.max_cnstits)
                total_cns.append(physics.jet_vars["n_cnts"].values)
                # break
            total_cns =np.concatenate(total_cns,0)
            print(max_cns)
            print(np.max(max_cns))
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            plot.plot_hist(total_cns, normalise=False, percentile_lst=[0,100],
                           log_yscale=True,ax=ax, dist_styles=[{"label": "Top jets"}],
                           style={"bins": 50})
            ax.set_ylabel("Entries")
            ax.set_xlabel("Number of constitutes")
            misc.save_fig(fig, "/home/users/a/algren/work/diffusion/plots/n_cnts_pr_top.pdf")
            sys.exit()
        else:
            physics = JetPhysics(test_path[:10],
                                 max_cnstits=175,
                                 max_ctxt_cnstits=400,
                                loader_config={"batch_size": 512,"num_workers": 4},
                                target_names=["eta", "phi", "pt"],
                                jet_scalars_cols=["eta", "phi", "pt", "mass", "mu", "n_cnts"],
                                pileup_path=PILEUP_PATH,
                                # datatype="image",
                                pileup_dist_args={"mu":200, "std":50}
                                )
        # physics.__getitem__(0)
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
            multi_data = {"images":[], "scalars":[], "images_ctxt":[],
                          "mask": [],
                          "mask_ctxt": []
                          }
            for i in tqdm(dataloader):
                multi_data["images"].append(i["images"])
                multi_data["mask"].append(i["mask"])
                
                multi_data["images_ctxt"].append(i["ctxt"]["cnts"])
                multi_data["mask_ctxt"].append(i["ctxt"]["mask"])
                
                multi_data["scalars"].append(i["ctxt"]["scalars"])
            # break

        for i in multi_data:
            multi_data[i] = T.concat(multi_data[i], 0).numpy()
            
        
        plt.figure()
        plt.hist(multi_data[i].sum(1), bins=50)

        # multi_data= np.load([i for i in ctxt_path if "soft" not in i][0], allow_pickle=True).item()

        # for i,name in zip(range(multi_data["scalars"].shape[1]), physics.jet_scalars_cols):
        #     plt.figure()
        #     plt.hist(multi_data["scalars"][:,i], bins=100)
        #     plt.title(name)

        # plot jets
        text_kwargs = dict(ha='center', va='center', fontsize=16,)
        legend_kwargs={"loc": "upper right", 'prop': {'size': 16}}
        for i,name, xlabel in zip(range(multi_data["scalars"].shape[1]),
                          physics.jet_scalars_cols,
                          [r"$\eta$", r"$\phi$",
                        #   [r"$\eta_{truth}-\eta_{Obs.}$", r"$\phi_{truth}-\phi_{Obs.}$",
                           r"$p_{\mathrm{T}}[\mathrm{GeV}]$", "Mass [GeV]",
                           r"$\mu$", "Number of cnsts."]):
            
            if (name in ["eta", "phi"]) & False:
                args = [(multi_data["scalars"][:,i]-physics.jet_vars[name])]
                percentile_lst=[0.101,99.9]
                name+="_diff"
                dist_styles=[{"label":"Top jet - Obs. jet", "color": "red"}]
                style={"bins":50}
            # elif (name in ["eta", "phi"]) & True:
            #     args = [(multi_data["scalars"][:,i], physics.jet_vars[name])]
            #     percentile_lst=[0.101,99.9]
            #     dist_styles=[{"label":"Obs. jet", "color": "red"},
            #                 {"label":"Top jet", "color": "black"}]
            #     style={"bins":50}
            elif "n_cnts" in name:
                args = [multi_data["mask_ctxt"].sum(1), physics.jet_vars[name]]
                style={"bins":50, "range": [0, 400]}
            else:
                args = [multi_data["scalars"][:,i]]
                percentile_lst=[0,100]
                dist_styles=[{"label":"Obs. jet", "color": "red"},
                            {"label":"Top jet", "color": "black"}]
                if name in physics.jet_vars:
                    # args = [(multi_data["scalars"][:,i]-physics.jet_vars[name])/physics.jet_vars[name]]
                    args.append(physics.jet_vars[name])
                else:
                    continue
                style={"bins":50}
                if "mass" in name:
                    style["range"]=[0, 1000]

            fig, ax = plt.subplots(1,1, figsize=(8,6))


            plot.plot_hist(*args, style=style, dist_styles=dist_styles,
                           ax=ax, log_yscale="GeV" in xlabel, percentile_lst=percentile_lst,
                           legend_kwargs=legend_kwargs)
            ax.set_xlabel(xlabel)
            if "eta" in name:
                ax.set_ylim(0, 0.04)
            plt.text(0.85,0.80, r"$\mu$~$\mathcal{N}(200, 50)$",
                     transform=ax.transAxes, **text_kwargs)
            if save_fig:
                misc.save_fig(fig, f"/home/users/a/algren/work/diffusion/plots/jet_var_{name}.pdf")

        # plot cnsts
        for i,name, xlabel in zip(range(multi_data["images_ctxt"][multi_data["mask_ctxt"]].shape[1]),
                          physics.jet_scalars_cols, [r"$\Delta \eta$", r"$\Delta \phi$",
                                                     r"$p_{\mathrm{T}}[\mathrm{GeV}]$"]):
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            
            if "pt" in name:
                args = [np.exp(multi_data["images_ctxt"][multi_data["mask_ctxt"]][:,i]),
                        np.exp(multi_data["images"][multi_data["mask"]][:,i])]
                bins = np.logspace(0,2, 100)*2-1
                style={"bins":bins}
                ax.set_yscale("log")
                ax.set_xscale("log")
            if name in ["eta", "phi"]:
                args = [multi_data["images_ctxt"][multi_data["mask_ctxt"]][:,i],
                        multi_data["images"][multi_data["mask"]][:,i]]
                style={"bins":100, "range":[-1.05,1.05]}
            
            plot.plot_hist(*args, style=style,
                           dist_styles=[{"label":"Obs cnsts.", "color": "red"},
                                        {"label":"Top cnsts.", "color": "black"}],
                           legend_kwargs=legend_kwargs,
                           ax=ax)
            ax.set_xlabel(xlabel)
            plt.text(0.85,0.80, r"$\mu$~$\mathcal{N}(200, 50)$",
                     transform=ax.transAxes, **text_kwargs)
            if "pt" in name:
                ax.set_xticks([1, 5, 10, 50, 100, 200])
            if save_fig:
                misc.save_fig(fig, f"/home/users/a/algren/work/diffusion/plots/cnts_var_{name}.pdf")
        sys.exit()
        
        # if "mask" in multi_data:
        #     added_cnts = multi_data["mask_ctxt"].sum(1)-multi_data["mask"].sum(1)
        #     plt.hist(added_cnts, bins=40)
        #     plt.xlabel("Number of constituents to the pc")
        #     sys.exit()
        # # Combine image channels for each variable
        # bin_range = [[-2.6, 2.6], [-2.6, 2.6]]
        # bins = 30
        # nr = 2
        # for i,j in [["images", "mask"], ["images_ctxt", "mask_ctxt"]]:
        #     plt.figure(figsize=(8*1.5,6*1.5))
        #     cnts = multi_data[i][nr][multi_data[j][nr]]
        #     print(f"Tracks: {multi_data[j][nr].sum()}")
        #     plt.hist2d(
        #         x=cnts[..., 0],
        #         y=cnts[..., 1],
        #         weights=cnts[..., -1],
        #         bins=bins,
        #         range=bounds,
        #     )
        #     plt.xlabel(r"$\Delta \eta$", fontsize=15)
        #     plt.ylabel(r"$\Delta \phi$", fontsize=15)
        # cnts = multi_data["images"]
        # data=[]
        # for i in tqdm(range(len(cnts))):
        #     images = np.histogram2d(multi_data["images"][i, ...,0],multi_data["images"][i, ...,1], weights=multi_data["images"][i, ...,2],
        #                             range = [[-2.6, 2.6], [-2.6, 2.6]], bins = 30)

        #     data.append(images)
        for i in range(5):
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(multi_data["images"][i].T,norm="log")
            ax[1].imshow(multi_data["images_ctxt"][i].T, norm="log")
            # fig, ax = plt.subplots(1,1)
            # ax.imshow(multi_data["images"][i].T-multi_data["images_ctxt"][i].T)
        # pileup_dist = PileupDist(**{"mu": 50, "std": 10})
        plt.figure()
        plt.hist(multi_data["scalars"][:, -1], 81)
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
