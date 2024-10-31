import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import numpy.lib.recfunctions as rfn

from src.jet_substructure import Substructure

import pandas as pd
import pyjet

import argparse

def _get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--input')
    parse.add_argument('--output')
    parse.add_argument('-N','--njets',type=int,default=None)
    return parse.parse_args()
    

def main():
    '''
    Take an input file of jets with constituents in the form pt,eta,phi,m and dump the substructure
    '''

    args = _get_args()
    
    outputjets = np.load(args.input)

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
            "mass": [],
            "pt": [],
            "eta": [],
            "phi": [],
    }

    for jet in outputjets[:args.njets]:
        incl_cluster = pyjet.cluster(rfn.unstructured_to_structured(jet.astype(np.float64),names=['pt','eta','phi','m']), R=1.0, p=-1)
        incl_jets = incl_cluster.inclusive_jets()[0]
        subs = Substructure(incl_jets, R=1.0)
    
        # NSubjettiness
        tau1 = subs.tau1()
        tau2 = subs.tau2()
        tau3 = subs.tau3()
        hlvs_dict['tau_1'].append(tau1)
        hlvs_dict['tau_2'].append(tau2)
        hlvs_dict['tau_3'].append(tau3)
        hlvs_dict['tau_21'].append(tau2 / tau1)
        hlvs_dict['tau_32'].append(tau3 / tau2)
        
        # Energy Splitting Functions
        hlvs_dict['d12'].append(subs.d12())
        hlvs_dict['d23'].append(subs.d23())
        
        # Energy Correlation Functions (first is simply the sum_pt of the jet)
        ecf2 = subs.ecf2()
        ecf3 = subs.ecf3()
        hlvs_dict['ecf2'].append(ecf2)
        hlvs_dict['ecf3'].append(ecf3)
        
        # ATLAS D2
        hlvs_dict['d2'].append(ecf3 / ecf2**3)
        
        #4vec
        hlvs_dict['pt'].append(incl_jets.pt)
        hlvs_dict['eta'].append(incl_jets.eta)
        hlvs_dict['phi'].append(incl_jets.phi)
        hlvs_dict['mass'].append(incl_jets.mass)

    df = pd.DataFrame.from_dict(hlvs_dict)
    df.to_hdf(args.output,"scalars")


if __name__=='__main__':
    main()
