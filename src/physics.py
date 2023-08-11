# from https://gitlab.cern.ch/mleigh/jetdiffusion/-/blob/matt_dev/src/physics.py
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import torch as T
import pandas as pd
import random

from src.utils import undo_log_squash
from src.eval_utils import EvaluateFramework
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
    
    
    return (sample, jet_vars)

class JetPhysics(EvaluateFramework):
    def __init__(self, jet_path:str, pileup_path:str=None):
        self.jet_path=jet_path
        self.pileup_path=pileup_path
        self.col_cnts = [f"cnts_{i}"for i in ["eta", "phi", "pt"]]
        self.col_jets = [f"jet_{i}"for i in ["eta", "phi", "pT", "mass"]]
        self.hist_kwargs={"percentile_lst":[0, 99],
                     "style": {"bins": 40, "histtype":"step"},
                     "dist_styles":[
                        {"marker":"o", "color":"black", "label":"Truth", "linewidth":0},
                        {"linestyle": "dotted","color":"blue", "label":"Generated", "drawstyle":"steps-mid"}]
                     }
        
        
        if isinstance(jet_path, (list, str)):
            (self.cnts_vars, self.mask_cnts, self.min_cnstits, self.max_cnstits,
             _, _, self.n_pc) = self.load_csv(self.jet_path)
        elif isinstance(jet_path, pd.DataFrame):
            (self.cnts_vars, self.mask_cnts, self.min_cnstits, self.max_cnstits,
             _, _, self.n_pc) = prepare_jet(jet_path)
        else:
            self.cnts_vars, self.mask_cnts = self.jet_path

        self.cnts_vars, self.jet_vars = self.physics_properties(self.cnts_vars,
                                                                 self.mask_cnts)

        if isinstance(pileup_path, str):
            self.pileup_cnts = self.load_csv(self.pileup_path)
            
        self.cnts_vars_rel = self.relative_pos(self.cnts_vars.copy(),
                                               self.jet_vars)
    
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

    def get_diffusion_data(self):
        return (self.cnts_vars_rel, self.mask_cnts, self.jet_vars.values,
                self.min_cnstits, self.max_cnstits, self.n_pc)
                
    def load_csv(self, paths):
        df = [pd.read_csv(i, dtype=np.float32) for i in paths]
        df = pd.concat(df, axis=0)
        df=df.rename(columns={i:i.replace(" ", "") for i in df.columns})
        return prepare_jet(df)

    def physics_properties(self, sample, mask):
        # calculate eta/phi/pT and jet variables
        return jet_variables(sample, mask)
    
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
    PATH = ["/home/users/a/algren/scratch/diffusion/pileup/ttbar_processed.csv"]
    
    physics = JetPhysics(PATH)
    mass = physics.jet_vars["mass"]
    mask = physics.jet_vars["pT"]>450
    plt.hist(mass[mask], bins=50, range=[0, 300])
    plt.xlabel("mass")
    plt.figure()
    plt.hist(physics.jet_vars["pT"][mask], bins=50,  range=[250, 750])
    plt.xlabel("pT")