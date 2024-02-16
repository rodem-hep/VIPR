"""Module with functions related to calculating jet substructure.

All of these are essentially based around pyjet.PseudoJet objects and/or
their constituent particles and calculates some of the most popular
high-level substructure observables.
"""

from pathlib import Path
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyjet
from dotmap import DotMap
from tqdm import tqdm
import pandas as pd

from src.physics import numpy_locals_to_mass_and_pt


def dij(
    pt1: Union[float, np.float64, np.ndarray],
    pt2: Union[float, np.float64, np.ndarray],
    drs: Union[float, np.float64, np.ndarray],
    radius_par: float = 1.0,
    exp: int = 1,
) -> Union[float, np.float64, np.ndarray]:
    """Calculates pairs of dij values as implemented in jet cluster algos.

    Takes two unstructued ndarrays of transverse momenta, an ndarray of delta_r
    distance between object pairs and calculates the dij as used in the kt jet
    clustering algorithm. radius_par is the clustering radius parameter, exp the
    exponent (exp=1 kt, exp=0 C/A, exp=-1 anti-kt).

    Expects pt1, pt2 and drs to be of the same length.

    Args:
        pt1: First unstructured ndarray of pT values.
        pt2: Second unstructured ndarray of pT values.
        drs: Unstructured ndarray of delta_r values.
        radius_par: The clustering radius parameter. Default is radius_par=1.0.
        exp: The exponent used in the clustering. Default is exp=1.

    Returns:
        Unstructured array of dij values for objects pairs.
    """

    if type(pt1) is not type(pt2) or type(pt1) is not type(drs):
        raise TypeError("Inputs 'pt1', 'pt2' and 'drs' are not of the same type")
    if not isinstance(pt1, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if (
        isinstance(pt1, np.ndarray)
        and isinstance(pt2, np.ndarray)
        and isinstance(drs, np.ndarray)
    ):
        if pt1.ndim != pt2.ndim or pt1.ndim != drs.ndim or pt1.ndim != 1:
            raise TypeError("Dimensions of input arrays are not equal to 1")
        if len(pt1) != len(pt2) or len(pt1) != len(drs):
            raise TypeError("Lengths of input arrays do not match")

    min_pt = np.amin((np.power(pt1, 2 * exp), np.power(pt2, 2 * exp)))
    return min_pt * drs * drs / radius_par / radius_par


def delta_r(
    eta1: Union[float, np.float64, np.ndarray],
    eta2: Union[float, np.float64, np.ndarray],
    phi1: Union[float, np.float64, np.ndarray],
    phi2: Union[float, np.float64, np.ndarray],
) -> Union[float, np.float64, np.ndarray]:
    """Calculates delta_r values between given ndarrays.

    Calculates the delta_r between objects. Takes unstructed ndarrays (or
    scalars) of eta1, eta2, phi1 and phi2 as input. Returns in the same format.
    This function can either handle eta1, phi1 to be numpy arrays (and eta2,
    phi2 to be floats), eta2, phi2 to be numpy arrays (and eta1, phi1 to be
    floats), or all four to be floats, and all four to be numpy arrays.
    Whenever numpy arrays are involved, they must be one-dimensional and of the
    same length.

    Args:
        eta1: First unstructured ndarray of eta values.
        eta2: Second unstructured ndarray of eta values.
        phi1: First unstructured ndarray of phi values.
        phi2: Second unstructured ndarray of phi values.

    Returns:
        Unstructured array of delta_r values between the pairs.
    """
    if type(eta1) is not type(phi1):
        raise TypeError("Inputs 'eta1' and 'phi1' must be of the same type")
    if type(eta2) is not type(phi2):
        raise TypeError("Inputs 'eta2' and 'phi2' must be of the same type")
    if not isinstance(eta1, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if not isinstance(eta2, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(eta1, np.ndarray) and isinstance(phi1, np.ndarray):
        if eta1.ndim != 1 or phi1.ndim != 1:
            raise TypeError("Dimension of 'eta1' or 'phi1' is not equal to 1")
        if len(eta1) != len(phi1):
            raise TypeError("Lengths of 'eta1' and 'phi1' do not match")
    if isinstance(eta2, np.ndarray) and isinstance(phi2, np.ndarray):
        if eta2.ndim != 1 or phi2.ndim != 1:
            raise TypeError("Dimension of 'eta2' or 'phi2' is not equal to 1")
        if len(eta2) != len(phi2):
            raise TypeError("Lengths of 'eta2' and 'phi2' do not match")
    if (
        isinstance(eta1, np.ndarray)
        and isinstance(eta2, np.ndarray)
        and len(eta1) != len(eta2)
    ):
        raise TypeError(
            "If 'eta1', 'eta2', 'phi1', 'phi2' are all of type np.ndarray, "
            "their lengths must be the same"
        )

    deta = np.absolute(eta1 - eta2)
    dphi = np.absolute(phi1 - phi2) % (2 * np.pi)
    dphi = np.min([2 * np.pi - dphi, dphi], axis=0)
    return np.sqrt(deta * deta + dphi * dphi)


def delta_r_min_to_axes(eta, phi, jet_axes):
    """Returns delta_r to closest jet axis for given eta/phi pair.

    Given unstructured ndarrays of eta and phi values and an unstructured
    ndarray of possible jet axes, finds the closest jet axis for each eta/phi
    pair in delta_r. Return the smallest delta_r value to any given axis, i.e.
    min(delta_r(cnst, axis1), delta_r(cnst, axis2), ...). Expects the jet_axes
    object to be of data type (_, eta, phi, _) per row.

    Args:
        eta: Unstructured ndarray of eta values.
        phi: Unstructured ndarray of phi values.
        jet_axes: Unstructured ndarray of jet axes, where eta and phi must be at
          index 1 and 2 per row, respectively (which corresponds to pyjet
          format).

    Returns:
        Unstructured ndarray with smallest delta_r obtained for each eta/phi pair.
    """
    if not isinstance(jet_axes, np.ndarray):
        raise TypeError("'jet_axes' needs to be of type np.ndarray")
    if jet_axes.ndim != 2:
        raise TypeError("np.ndarray 'jet_axes' needs to have dimension 2")
    if jet_axes.shape[1] != 4:
        raise TypeError("'jet_axes' needs to be of length '4' along the second axis")
    if type(eta) is not type(phi):
        raise TypeError("Inputs 'eta' and 'phi' must be of the same type")
    if not isinstance(eta, (float, np.float64, np.ndarray)):
        raise TypeError("Inputs must be of type 'float', 'np.float64' or 'np.ndarray'")
    if isinstance(eta, np.ndarray) and isinstance(phi, np.ndarray):
        if eta.ndim != 1 or phi.ndim != 1:
            raise TypeError("Dimension of 'eta' or 'phi' is not equal to 1")
        if len(eta) != len(phi):
            raise TypeError("Lengths of 'eta' and 'phi' do not match")
    delta_r_list = np.array([delta_r(eta, axis[1], phi, axis[2]) for axis in jet_axes])
    return np.amin(delta_r_list, axis=0)


class Substructure:
    """Calculates and holds substructure information per jet.

    This class calculates substructure observables for a
    pyjet.PseudoJet. Takes a PseudoJet object as input, calculates some
    essential re-clustering in the init function, then allows to
    retrieve various sorts of substructure variables through accessors.
    These are only calculated when called.
    """

    def __init__(self, jet, R):
        """Calculate essential reclustering for given pyjet.PseudoJet object.

        This retrieves the constituent particles of the jet, reshapes them into
        an array with axis0 = n_cnsts and axis1 = (pt, eta, phi, mass). Then
        reclusters constituent particles with the kt algorithm and stores lists
        of N-exclusive jets (exclusive kt clustering). Falls back to (N-1)
        clustering if there are not enough constituent particles for N.

        Args:
            jet: The pyjet.PseudoJet object.
            R: The jet radius used for reclustering.
        """
        R = 1.0 if R is None else R
        self._cnsts = jet.constituents_array()
        self._cnsts = self._cnsts.view(dtype=np.float64).reshape(
            self._cnsts.shape + (-1,)
        )

        rclst = pyjet.cluster(jet.constituents_array(), R=R, p=1)
        self._subjets1 = np.array(
            [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(1)]
        )
        try:
            self._subjets2 = np.array(
                [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(2)]
            )
        except ValueError:
            self._subjets2 = self._subjets1
        try:
            self._subjets3 = np.array(
                [[_j.pt, _j.eta, _j.phi, _j.mass] for _j in rclst.exclusive_jets(3)]
            )
        except ValueError:
            self._subjets3 = self._subjets2

        # Store the frequently used sum of constituent transverse momenta.
        self._ptsum = np.sum(self._cnsts[:, 0])

    def d12(self):
        """Calculates the d12 splitting scale.

        Calculates the splitting scale for 2-jet exclusive clustering:
        one expects one of the jets in N-exclusive clustering to split
        in two in N+1-exclusive clustering. Locates these two 'new' jets
        and returns the square root of their d_ij. If something goes
        wrong, default to 0.
        """
        cmpl_indices = np.nonzero(
            np.isin(self._subjets2[:, 0], self._subjets1[:, 0], invert=True)
        )[0]
        if not len(cmpl_indices) == 2:
            return 0.0

        _j1 = self._subjets2[cmpl_indices[0]]
        _j2 = self._subjets2[cmpl_indices[1]]
        distance = dij(_j1[0], _j2[0], delta_r(_j1[1], _j2[1], _j1[2], _j2[2]))
        return 1.5 * np.sqrt(distance)

    def d23(self):
        """Calculates the d23 splitting scale.

        Calculates the splitting scale for 3-jet exclusive clustering:
        one expects one of the jets in N-exclusive clustering to split
        in two in N+1-exclusive clustering. Locates these two 'new' jets
        and returns the square root of their d_ij. If something goes
        wrong, default to 0.
        """
        cmpl_indices = np.nonzero(
            np.isin(self._subjets3[:, 0], self._subjets2[:, 0], invert=True)
        )[0]
        if not len(cmpl_indices) == 2:
            return 0.0

        _j1 = self._subjets3[cmpl_indices[0]]
        _j2 = self._subjets3[cmpl_indices[1]]
        distance = dij(_j1[0], _j2[0], delta_r(_j1[1], _j2[1], _j1[2], _j2[2]))
        return 1.5 * np.sqrt(distance)

    def ecf2(self):
        """Calculates the degree-2 energy correlation factor.

        Calculates the degree-2 energy correlation factor of the constituent
        particles. Takes transverse momenta and delta_r distances into account:

        >> sum (i<j in cnsts) pt(i) * pt(j) * delta_r(i, j)

        To avoid for-loop nesting, creates an array of unique index pairs, then
        uses those to access the constituent particles to be able to vectorise
        the operation. Internal function calc_ecf2(i, j) takes lists of
        components.
        """
        indices = np.arange(len(self._cnsts), dtype=np.uint8)
        idx_pairs = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)
        idx_pairs = idx_pairs[(idx_pairs[:, 0] < idx_pairs[:, 1])]

        def calc_ecf2(i, j):
            return i[:, 0] * j[:, 0] * delta_r(i[:, 1], j[:, 1], i[:, 2], j[:, 2])

        return (
            calc_ecf2(self._cnsts[idx_pairs][:, 0], self._cnsts[idx_pairs][:, 1]).sum()
            / self._ptsum
            / self._ptsum
        )

    def ecf3(self):
        """Calculates the degree-3 energy correlation factor.

        Calculates the degree-3 energy correlation factor of the constituent
        particles. Takes transverse momenta and delta_r distances into account:

        >> sum (i<j<k in cnsts) pt(i) * pt(j) * pt(k)
        >>                      * delta_r(i, j) * delta_r(j, k) * delta_r(k, i)

        To avoid for-loop nesting, creates array of unique index triplets, then
        uses those to access the constituent particles to be able to vectorise
        the operation. Internal function calc_ecf2(i, j, k) takes lists of
        components.
        """
        indices = np.arange(len(self._cnsts), dtype=np.uint8)
        idx_pairs = np.array(np.meshgrid(indices, indices, indices)).T.reshape(-1, 3)
        idx_pairs = idx_pairs[
            (idx_pairs[:, 0] < idx_pairs[:, 1]) & (idx_pairs[:, 1] < idx_pairs[:, 2])
        ]

        def calc_ecf3(i, j, k):
            return (
                i[:, 0]
                * j[:, 0]
                * k[:, 0]
                * delta_r(i[:, 1], j[:, 1], i[:, 2], j[:, 2])
                * delta_r(j[:, 1], k[:, 1], j[:, 2], k[:, 2])
                * delta_r(k[:, 1], i[:, 1], k[:, 2], i[:, 2])
            )

        return (
            calc_ecf3(
                self._cnsts[idx_pairs][:, 0],
                self._cnsts[idx_pairs][:, 1],
                self._cnsts[idx_pairs][:, 2],
            ).sum()
            / self._ptsum
            / self._ptsum
            / self._ptsum
        )

    def tau1(self):
        """Calculates the 1-subjettiness.

        Calculates the 1-subjettiness (sum over minimal distances to jet
        axes for exclusive 1-jet clustering, weighted with constituent
        particle pT). Returns the dimensionless version of tau1, i.e.,
        divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(
            self._cnsts[:, 1], self._cnsts[:, 2], self._subjets1
        )
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum

    def tau2(self):
        """Calculates the 2-subjettiness.

        Calculates the 2-subjettiness (sum over minimal distances to jet
        axes for exclusive 2-jet clustering, weighted with constituent
        particle pT). Returns the dimensionless version of tau2, i.e.,
        divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(
            self._cnsts[:, 1], self._cnsts[:, 2], self._subjets2
        )
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum

    def tau3(self):
        """Calculates the 3-subjettiness.

        Calculates the 3-subjettiness (sum over minimal distances to jet
        axes for exclusive 3-jet clustering, weighted with constituent
        particle pT). Returns the dimensionless version of tau3, i.e.,
        divided by the sum of all constituent transverse momenta.
        """
        dr_vals = delta_r_min_to_axes(
            self._cnsts[:, 1], self._cnsts[:, 2], self._subjets3
        )
        return np.sum(self._cnsts[:, 0] * dr_vals) / self._ptsum


def dump_hlvs(
    jets,
    masks, 
    # rel_jet:
    # file_path: Path,
    out_path: Path,
    R: float = 1,
    p: float = -1.0,
    # plot: bool = False,
    addi_col: dict=None
) -> None:
    """Given the nodes of a point cloud jet, compute the subtstructure
    variables and dump them to a file."""
    if addi_col is None:
        addi_col={}
    # First load the pt and mass, variables often needed alongside substructure
    # outputs jet_px, jet_py, jet_pz, jet_eta, jet_phi, jet_pt, jet_m
    pt_mass = numpy_locals_to_mass_and_pt(jets, masks)
    pt = pt_mass[:, -2]
    mass = pt_mass[:, -1]

    # Get the scalar sum of the pts (this will be the ecf1 variable)
    sum_pts = np.sum(jets[..., -1], axis=-1)

    # The substructure functions need data to be in [pt, eta, phi, m]
    jets = np.concatenate(
        [jets[..., [2, 1, 0]], np.zeros(shape=(*jets.shape[:-1], 1))],
        axis=-1,
    )
    jets = np.ascontiguousarray(jets)

    # Initialise the lists to hold the substructure variables
    hlvs_dict = {
            "tau_1": [],
            "tau_2": [],
            "tau_3": [],
            "tau_21": [],
            "tau_32": [],
            "d12": [],
            "d23": [],
            "ecf2": [],
            "ecf3": [],
            "d2": [],
            "pt": pt,
            "mass": mass,
            "sum_pts": sum_pts,
        }
    
    hlvs_dict.update(addi_col)
    
    hlvs = DotMap(hlvs_dict)

    for jet, mask in tqdm(
        zip(jets, masks),
        total=len(jets),
        miniters=len(jets) // 20,
        desc="Computing substructure variables",
    ):

        if mask.sum() < 1:
            print("Found jet with 0 constituents! Skipping.")
            continue

        # pyjet needs each jet to be a be a structured array of type float64
        jet = jet[mask].view(
            [
                ("pt", np.float64),
                ("eta", np.float64),
                ("phi", np.float64),
                ("mass", np.float64),
            ]
        )
        incl_cluster = pyjet.cluster(jet, R=R, p=p)
        incl_jets = incl_cluster.inclusive_jets()[0]
        subs = Substructure(incl_jets, R=R)

        # NSubjettiness
        tau1 = subs.tau1()
        tau2 = subs.tau2()
        tau3 = subs.tau3()
        hlvs.tau_1.append(tau1)
        hlvs.tau_2.append(tau2)
        hlvs.tau_3.append(tau3)
        hlvs.tau_21.append(tau2 / tau1)
        hlvs.tau_32.append(tau3 / tau2)

        # Energy Splitting Functions
        hlvs.d12.append(subs.d12())
        hlvs.d23.append(subs.d23())

        # Energy Correlation Functions (first is simply the sum_pt of the jet)
        ecf2 = subs.ecf2()
        ecf3 = subs.ecf3()
        hlvs.ecf2.append(ecf2)
        hlvs.ecf3.append(ecf3)

        # ATLAS D2
        hlvs.d2.append(ecf3 / ecf2**3)
    
    if False:
        # Save all the data to an HDF file
        with h5py.File(out_path, mode="w") as file:
            for k, v in hlvs.items():
                file.create_dataset(k, data=v)
    else:
        hlvs_dict = dict(hlvs)
        
        # to remove features with incorrect length
        not_correct_len = [i for i,j in hlvs_dict.items()
                           if len(j) != len(hlvs_dict["mass"])]
        
        print(f"Features with incorrect length: {not_correct_len}")

        for i in not_correct_len:
            hlvs_dict.pop(i)

        # Save all the data to an HDF file with pandas
        pd.DataFrame.from_dict(hlvs_dict).to_hdf(out_path, key='df', mode='w')
