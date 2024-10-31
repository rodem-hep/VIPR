import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)
import os
import numpy as np
import numpy.lib.recfunctions as rfn
from tqdm import tqdm
import fastjet
import energyflow

import argparse

def ptetaphim_to_pxpypzE(jet):
    '''Input fourvector (pt,eta,phi,m), return fourvector (px,py,pz,E)'''
    px = jet[...,0]*np.cos(jet[...,2])
    py = jet[...,0]*np.sin(jet[...,2])
    pz = jet[...,0]*np.sinh(jet[...,1])
    E  = np.sqrt((jet[...,0]*np.cosh(jet[...,1]))**2 + jet[...,3]**2)
    return np.dstack([px,py,pz,E])

def _get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--input', default='/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/obs_jets/obs_jet_pileup_mu_200_std_50_size_99990_10_07_2024_11_26_49.npy')
    parse.add_argument('--output', default='/srv/beegfs/scratch/groups/rodem/pileup_diffusion/data/softdrop')
    parse.add_argument('-z','--zcut',type=float,default=0.1)
    parse.add_argument('-b','--beta',type=float,default=0)
    parse.add_argument('-N','--njets',type=int,default=None)
    return parse.parse_args()


def main():
    '''
    Load a file of jets with constituents in the Malte format of eta,phi,pt, turn them back into pt,eta,phi,m, then run fastjet on top and produce softdrop jets
    '''

    args = _get_args()

    #make sure the output directory exists
    save_path = '/'.join(args.output.split('/')[:-1])
    if os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    data = np.load(args.input, allow_pickle=True).item()    
    # jconst = np.dstack([data['cnts'][:],np.sqrt(np.sum(data['cnts'][:]**2,axis=-1))[...,np.newaxis]]).astype(np.float64)
    jconst = np.dstack([data['cnts'][:],np.zeros(data['cnts'].shape[:-1])[...,np.newaxis]]).astype(np.float64)
    jconst = jconst [:,:,[2,0,1,3]] # pt, eta, phi, m
    jconst = ptetaphim_to_pxpypzE(jconst) # fastjet python bindings needs px, py, pz, E

    # We use anti-kt R=1 as standard, could make this an option. fastjet.cambridge_algorithm could be another (but old) option. R=0.8 for CMS friends
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm,1.0)

    maxjets = args.njets

    outputjets = np.zeros_like(jconst)[:maxjets]

    for i,jet in tqdm(enumerate(jconst[:maxjets]),total=len(jconst)):

        psjet = [fastjet.PseudoJet(*j) for j in jet if j.any()]
        cluster_seq = fastjet.ClusterSequence(psjet,jetdef)
        
        # njets = cluster_seq.exclusive_jets(1)## To work out why this isn't working
        njets = cluster_seq.inclusive_jets(ptmin=25) ### could be a problem if we were ever to have jets with pt < 25 GeV or it decided to split the jet into N jets
        
        # Just in case, get the highest pt jet. Don't want to end up with 10 jets with 0 pt...
        pts = [je.pt() for je in njets]
        jet = njets[np.argmax(pts)]

        # Run softdrop through energyflow package because fastjet inbuilt python binding has issues
        sd_jet = energyflow.softdrop(jet,zcut=args.zcut,beta=args.beta, R=1)
        consts = sd_jet.constituents()

        # Save jets in standard form pt,eta,phi,m (where m = 0 for all constituents)
        outputjets[i,:len(consts)] = np.concatenate([np.array([j.pt(),j.eta(),j.phi(),0])[np.newaxis]\
                                                     for j in consts],axis=0)

    np.save(args.output,outputjets)

if __name__=='__main__':
    main()
