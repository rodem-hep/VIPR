# from https://gitlab.cern.ch/mleigh/jetdiffusion/-/blob/matt_dev/src/physics.py
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from typing import Union

import numpy as np
import torch as T
import pandas as pd
from torch.utils.data import Dataset,DataLoader


from src.utils import undo_log_squash, log_squash
from src.eval_utils import EvaluateFramework
from src.prepare_data import fill_data_in_pc, matrix_to_point_cloud
from tools.transformations import log_squash, undo_log_squash

def calculate_pT(px, py):
    return np.sqrt(px**2+py**2)

def calculate_phi(px, pT):
    return np.arccos(px/pT)

def calculate_eta(pz, pT):
    return np.arcsinh(pz/pT)

def detector_dimensions(df:np.ndarray):

    # calculate pT
    pT = calculate_pT(df[..., 0], df[..., 1])

    # calculate phi
    phi = calculate_phi(df[..., 0], pT)

    # calculate eta
    eta = calculate_eta(df[..., 2], pT)
    
    return eta[..., None], phi[..., None], pT[..., None]

def jet_variables(sample, mask):
    # calculate eta/phi/pT
    # eta, phi, pT = detector_dimensions(sample)

    # sample_epp = np.concatenate([eta, phi, pT], -1)
    
    # calculate summary jet features
    jet_vars = numpy_locals_to_mass_and_pt(sample, mask)
    jet_vars = pd.DataFrame(jet_vars, columns=["eta", "phi", "pT", "mass"])
    
    
    return jet_vars

class JetPhysics(EvaluateFramework, Dataset):
    # TODO dangerous that it is both a eval framework and dataloader
    def __init__(self, jet_path:Union[str, list], pileup_path:Union[str, list]=None, **kwargs):
        self.jet_path=jet_path
        self.standardize_bool = kwargs.get("standardize_bool", False)
        self.loader_config = kwargs.get("loader_config", {})
        self.max_pileup_size = kwargs.get("max_pileup_size", {})
        self.pileup_path=pileup_path
        self.col_jets = [f"jet_{i}"for i in ["eta", "phi", "pT", "mass"]]
        self.hist_kwargs={"percentile_lst":[0, 99],
                     "style": {"bins": 40, "histtype":"step"},
                     "dist_styles":[
                        {"marker":"o", "color":"black", "label":"Truth", "linewidth":0},
                        {"linestyle": "dotted","color":"blue", "label":"Generated", "drawstyle":"steps-mid"}]
                     }
        
        # get jet data
        (self.cnts_vars, self.mask_cnts, self.min_cnstits, self.max_cnstits, self.n_pc,
         self.col_cnts) = self.load_csv(self.jet_path)
        self.col_cnts = [f"cnts_{i}"for i in self.col_cnts]
        self.jet_vars = self.physics_properties(self.cnts_vars[..., -3:],self.mask_cnts)

        self.cnts_vars_rel = self.relative_pos(self.cnts_vars[..., -3:].copy(),
                                               self.jet_vars)

        # get jet data
        if pileup_path is not None:
            (self.pileup_cnts, self.pileup_mask_cnts, self.min_pileup_cnstits,
             self.max_pileup_cnstits, self.n_pileup_pc, self.col_pileup_cnts
             )  = self.load_csv(self.pileup_path)        
    

    def load_csv(self, paths):
        df = [pd.read_csv(i, dtype=np.float32) for i in paths]
        df = pd.concat(df, axis=0)
        df=df.rename(columns={i:i.replace(" ", "") for i in df.columns})
        event_nr_columns = [i for i in df.columns if "number" in i ]
        if all(np.in1d(["px", "py", "pz"],df.columns)):
            
            df["eta"],df["phi"],df["pt"]= detector_dimensions(df[["px", "py", "pz"]].values)

        cnts_df, mask = matrix_to_point_cloud(df.values, df[event_nr_columns].values)

        return cnts_df, mask, mask.sum(1).min(), mask.sum(1).max(), len(cnts_df), df.columns
    
    @staticmethod
    def relative_pos(cnts_vars, jet_vars, reverse=False):
        cnts_vars_rel = cnts_vars.copy()
        
        if reverse:
            for nr, i in enumerate(["eta", "phi"]):
                cnts_vars_rel[..., nr] += jet_vars[:, nr][:,None]

            cnts_vars_rel[..., nr+1] = undo_log_squash(cnts_vars_rel[..., nr+1])
        else:

            for nr, i in enumerate(["eta", "phi"]):
                cnts_vars_rel[..., nr] -= jet_vars[i].values[:,None]

            cnts_vars_rel[..., nr+1] = log_squash(cnts_vars_rel[..., nr+1])

        return cnts_vars_rel

                

    def physics_properties(self, sample, mask):
        # calculate eta/phi/pT and jet variables
        return jet_variables(sample, mask)
    
    def concat_pileup(self, idx_jet, n_pileup_to_select:int=100):
        # generate events with pileup
        cnts_vars = self.cnts_vars[idx_jet]
        mask_cnts_vars = self.mask_cnts[idx_jet]

        jet_vars = self.jet_vars.iloc[idx_jet:idx_jet+1]
        
        idx = np.random.choice(np.arange(0, self.n_pileup_pc,1), n_pileup_to_select)

        pileup_cnts = self.pileup_cnts[idx]
        pileup_cnts_mask = self.pileup_mask_cnts[idx]

        pileup_cnts_rel = self.relative_pos(pileup_cnts[...,-3:].copy(), jet_vars)

        delta_R_mask = np.sqrt(np.sum(pileup_cnts_rel[..., :2]**2, -1))<1

        pileup_mask = delta_R_mask & pileup_cnts_mask
        
        # concat and fill full event matrix
        # TODO should pply be shuffled at some point
        evt_w_pileup = np.concatenate([cnts_vars[mask_cnts_vars],
                                       pileup_cnts[pileup_mask]], 0)
        
        full_event = np.ones((self.max_pileup_size, cnts_vars.shape[-1]))*-999
        full_event[:len(evt_w_pileup)] = evt_w_pileup
        
        
        return full_event, np.any(full_event!=-999, -1)

    def get_ctxt_shape(self):
         # TODO not hardcode n features, add if statement for non-PC ctxt
        return {"pc": [self.max_pileup_size, 3]}
    
    def get_diffusion_data(self):
        return (self.cnts_vars_rel, self.mask_cnts, self.jet_vars.values,
                self.min_cnstits, self.max_cnstits, self.n_pc)

    def train_dataloader(self):
        return DataLoader(self, **self.loader_config)

    def test_dataloader(self):
        return DataLoader(self, **self.loader_config)

    def _shape(self):
        return [150, 3] # TODO should not be hardcoded!
    
    def __len__(self):
        return self.n_pc


    def __getitem__(self, idx):
        data = {"images": np.float32(self.cnts_vars[idx, ..., 3:6]), "mask": self.mask_cnts[idx]}
        # data = {"images": np.float32((self.sample[idx]-self.mean)/self.std), "mask": self.mask[idx]}

        if True: # TODO add proper boolean
            data["ctxt"] = {}
            event, mask = self.concat_pileup(idx)
            # data["ctxt"] = np.float32((self.ctxt[idx]-self.ctxt_mean)/self.ctxt_std)
            data["ctxt"]["pc"] = event[..., 3:6]
            data["ctxt"]["mask"] = mask

        return data
    
    def __call__(self, gen_data, mask, ctxt):
        log={}
        
        # renorm
        ctxt = ctxt*self.jet_vars.std(0).values+self.jet_vars.mean(0).values

        # reverse relative
        gen_cnts_vars = self.relative_pos(gen_data, ctxt, reverse=True)
        generated_jet_vars = numpy_locals_to_mass_and_pt(gen_cnts_vars, mask)

        log = self.plot_marginals(ctxt,
                                  generated_jet_vars,
                                  col_name=self.col_jets, hist_kwargs=self.hist_kwargs,
                                  log=log)

        log = self.plot_marginals(self.cnts_vars[self.mask_cnts][:100_000],
                                  gen_cnts_vars[mask],
                                  col_name=self.col_cnts, log=log,
                                  hist_kwargs=self.hist_kwargs)

        return log


def prepare_jet(df, dummy_val = -999):
    df = df.values
    df = df[:,1:]
    df = df.reshape(df.shape[0], df.shape[1] // 150, 150)
    df = np.moveaxis(df, -1,-2)
    mask = df!=dummy_val
    
    # calculate eta/phi/pT
    df = np.concatenate(detector_dimensions(df), -1)

    mean = np.array([df[:,:,i][mask[:,:,i]].mean()
                     for i in range(df.shape[2])])[None, :]
    
    std = np.array([df[:,:,i][mask[:,:,i]].std()
                    for i in range(df.shape[2])])[None, :]

    return (np.float32(df), np.all(mask,2), mask.sum(1).min(), mask.sum(1).max(),
            mean, std, len(df))

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
    jet_px = (pt * np.cos(phi) * mask).sum(axis=-1)
    jet_py = (pt * np.sin(phi) * mask).sum(axis=-1)
    jet_pz = (pt * np.sinh(eta) * mask).sum(axis=-1)
    jet_e = (pt * np.cosh(eta) * mask).sum(axis=-1)

    # Get the derived jet values, the clamps ensure NaNs dont occur
    jet_pt = np.sqrt(np.clip(jet_px**2 + jet_py**2, 0, None))
    jet_m = np.sqrt(np.clip(jet_e**2 - jet_px**2 - jet_py**2 - jet_pz**2, 0, None))
    
    # position of jet
    jet_phi = calculate_phi(jet_px, jet_pt)
    jet_eta = calculate_eta(jet_pz, jet_pt)

    return np.vstack([jet_eta, jet_phi, jet_pt, jet_m]).T

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    PATH = ["/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"]
    PILEUP_PATH = ["/home/users/a/algren/scratch/diffusion/pileup/pileup.csv"]
    
    physics = JetPhysics(PATH, pileup_path=PILEUP_PATH)
    
    dataloader = iter(DataLoader(physics))
    
    data = next(dataloader)
    
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
        mass = physics.jet_vars["mass"]
        mask = physics.jet_vars["pT"]>450
        plt.hist(mass[mask], bins=50, range=[0, 300])
        plt.xlabel("mass")
        plt.figure()
        plt.hist(physics.jet_vars["pT"][mask], bins=50,  range=[250, 750])
        plt.xlabel("pT")
        for i in range(3,6):
        # for i in range(physics.cnts_vars.shape[-1]):
            plt.figure()
            plt.hist(physics.cnts_vars[physics.mask_cnts][:,i], bins=50)
            # plt.hist(log_squash(physics.cnts_vars[physics.mask_cnts][:,i]), bins=50)
            plt.xlabel(physics.col_cnts[i])
            plt.yscale("log")
