import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import sys
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from typing import Mapping, Union
import h5py


import pandas as pd
import numpy as np

import torch as T
from torch.utils.data import Dataset,DataLoader
from tools import misc
import src.prepare_data as prepare_data 
from src.physics import JetPhysics
import hydra

class DataModule:
    def __init__(self, *,
                 train_set: partial, test_set: partial,
                 loader_config: Mapping, img_enc:partial=None,
                 run_img_enc_test:bool=True
                 ):

        self.img_enc=img_enc
        self.loader_config = loader_config
        self.train_sample=train_set()
        self.test_sample=test_set()
        self.run_img_enc_test=run_img_enc_test

        if img_enc is not None:
            self._img_enc = self.img_enc

    def _img_enc(self, dataloader):
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._img_enc(DataLoader(self.train_sample, **self.loader_config))
        # return DataLoader(self.train_sample, **self.loader_config)

    def test_dataloader(self) -> DataLoader:
        test_config = deepcopy(self.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        if self.run_img_enc_test:
            return self._img_enc(DataLoader(self.test_sample, **test_config))
        else:
            return DataLoader(self.test_sample, **test_config)
        
class ImageModule(Dataset):
    def __init__(self, data, img_enhancement=None):
        self.data=data
        self.img_enhancement=img_enhancement
        self._iter = 0

    def __len__(self):
        return len(self.data)
    
    def _shape(self):
        raise ValueError("not implemented yet")
        return self.sample.shape[1:]

    def __iter__(self):
        # image = self.data[idx]
        self._iter+=1
        print(self._iter)
        for image, _ in self.data:
            if self.img_enhancement is not None:
                yield image, self.img_enhancement(image.clone())
            else:
                yield image, None
                
class PointCloudModule:
    def __init__(self, *, train_path, test_path, standardize_bool, loader_config,
                 data_name:str=None):
        self.data_name=data_name
        self.train_path = train_path
        self.test_path = test_path
        self.loader_config=loader_config
        self.train_sample = PointCloud(self.train_path, standardize_bool=standardize_bool)
        self.test_sample = PointCloud(self.test_path,
                                      max_cnstits=self.train_sample.max_cnstits,
                                      standardize_bool=standardize_bool)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_sample, **self.loader_config)

    def test_dataloader(self) -> DataLoader:
        test_config = deepcopy(self.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        return DataLoader(self.test_sample, **test_config)
    
    def _shape(self):
        return self.train_sample._shape()

    def get_mean(self):
        return T.tensor(self.train_sample._get_mean())

    def get_std(self):
        return T.tensor(self.train_sample._get_std())

    def get_max_cnstits(self):
        return self.train_sample.max_cnstits

    def get_min_cnstits(self):
        return self.train_sample.min_cnstits
    
    def get_data_name(self):
        return self.data_name

class PCLoader(Dataset):
    def __init__(self, sample, mask, ctxt=None, run_norm:bool=False):
        self.sample=sample
        self.mask=mask
        self.ctxt = ctxt
        self.run_norm=run_norm
        self.mean = getattr(self, 'mean', np.zeros(self.sample.shape[-1],))
        self.std = getattr(self, 'std', np.ones(self.sample.shape[-1],))

        if (ctxt is not None) & self.run_norm:
            self.ctxt_mean = getattr(self, 'ctxt_mean', np.zeros(self.ctxt.shape[-1],))
            self.ctxt_std = getattr(self, 'ctxt_std', np.ones(self.ctxt.shape[-1],))

    def get_norm_data(self):
        return (self.sample-self.mean)/self.std

    def get_normed_ctxt(self):
        return (self.ctxt-self.ctxt_mean)/self.ctxt_std

    def _get_mean(self):
        
        return self.mean
        
    def _get_std(self):
        
        return self.std

    def __len__(self):
        return len(self.sample)
    
    def _shape(self):
        return self.sample.shape[1:]

    def __getitem__(self, idx):
        data = {"images": None, "mask": self.mask[idx]}
        if self.run_norm:
            _data = np.float32((self.sample[idx]-self.mean)/self.std)
        else:
            _data = self.sample[idx]
            
        data["images"] = _data
            
        if self.ctxt is not None:
            data["ctxt"] = {}
            if "cnts" in self.ctxt:
                data["ctxt"]["cnts"] = np.float32(self.ctxt["cnts"][idx])
                data["ctxt"]["mask"] = self.ctxt["mask"][idx] # TODO not normed
            if "scalars" in self.ctxt:
                data["ctxt"]["scalars"] = np.float32(self.ctxt["scalars"][idx])

        return data

class PointCloud(PCLoader):
    def __init__(self, paths, max_cnstits=None, standardize_bool:bool=False):
        self.paths = paths
        
        # used for non-padded data
        self.max_cnstits=max_cnstits
        self.standardize_bool=standardize_bool
        self.idx_number=None
        self.ctxt=None

        # load data
        self.load_data()
        
        # get norm values
        self.calculate_norms()
        
        super().__init__(self.sample, self.mask, self.ctxt)


    def load_data(self):

        # load data
        if ".csv" in self.paths[0]:
            self.df = [pd.read_csv(i, dtype=np.float32) for i in self.paths]
            self.df = pd.concat(self.df, axis=0)
            self.df=self.df.rename(columns={i:i.replace(" ", "") for i in self.df.columns})
        elif (".h5" in self.paths[0]) or ("hdf5" in self.path[0]):
            self.df = h5py.File(self.paths[0])


        # split and reshape
        # sample should be: (n_pc x features x constituents)
        if "mnist" in self.paths[0]:
            (self.sample, self.mask, self.min_cnstits, max_cnstits, self.n_pc
             ) = prepare_data.prepare_mnist(self.df, max_cnstits=self.max_cnstits)
        elif "pileup" in self.paths[0]:
            jet_data = JetPhysics(self.df)
            (self.sample, self.mask, self.ctxt, self.min_cnstits,
             max_cnstits, self.n_pc) = jet_data.get_diffusion_data()
        elif "shapenet" in self.paths[0]:
            (self.sample, self.mask, self.min_cnstits, max_cnstits, self.n_pc
             ) = prepare_data.prepare_shapenet(self.df)
        else:
            raise ValueError("Unknown data path")

        print(f"Sample size {self.sample.shape}")

        if self.max_cnstits is None:
            self.max_cnstits=max_cnstits

        self.pc_shape=self.sample.shape[1:]
        

    def calculate_norms(self):
        self.mean, self.std = (self.sample[self.mask, :].mean(0),
                               self.sample[self.mask, :].std(0))

        if self.ctxt is not None:
            if len(self.ctxt.shape)==2:
                self.ctxt_mean, self.ctxt_std = self.ctxt.mean(0), self.ctxt.std(0)
            else:
                raise NotImplementedError("ctxt for this shape not implemented yet")
            
            
        #norm sample
        if self.standardize_bool:
            self.sample = (self.sample-self.mean)/self.std
    
def generate_gaussian_noise(shape:list, datatype:str, 
                            n_constituents: Union[tuple, int]=None,
                            eval_ctxt:np.ndarray=None,
                            size:int=9, device:str="cuda"):
    "generate noisy images for diffusion"
    if "image" in datatype:
        return T.randn(
                tuple(size+shape),
                device=device
                ).float()
    elif "pc" in datatype:
        if n_constituents is None:
            raise ValueError("n_constituents has to be defined")
        
        # calculate the n constituents
        if isinstance(n_constituents, tuple):
            n_constituents = np.random.randint(*n_constituents, size)
        elif isinstance(n_constituents, int):
            n_constituents = np.random.randint(1, n_constituents, size=size)  
            
        if not isinstance(n_constituents, np.ndarray):
            raise TypeError("n_constituents has to be a np.array")

        mask = np.zeros([size]+shape[:1])==1
        for nr,i in enumerate(n_constituents[:size]):
            mask[nr, :i] = True
        
        return DataLoader(
            PCLoader(T.randn(tuple([size]+shape)).numpy(),mask,ctxt=eval_ctxt),
            batch_size=128, num_workers=4)

if __name__ == "__main__":
    # %matplotlib widget
    if False: # testing images
        import matplotlib.pyplot as plt
        config = misc.load_yaml("configs/data_cfg.yaml")
        data = hydra.utils.instantiate(config.train_set)
        dataloader = hydra.utils.instantiate(config.loader_cfg)(data)
        image_loader = hydra.utils.instantiate(config.img_enc)(dataloader)
        # downscale_func = T.nn.AvgPool2d(2)
        # image_loader = ImageEnhancement(dataloader, downscale_func)
        
        for i, cvx in image_loader:
            # print(i)
            break
        cvx= cvx.permute(0, 2, 3, 1)
        i= i.permute(0, 2, 3, 1)


        style = {"vmax":1, "vmin":0}
        figsize=(4*8, 6)
        _, ax = plt.subplots(1,4, figsize=figsize)
        for a, img in zip(ax, cvx):
            a.imshow(img, **style)

        _, ax = plt.subplots(1,4, figsize=figsize)
        for a, img in zip(ax, i):
            a.imshow(img, **style)
    elif True: # testing shapenet pc
        import h5py
        h5_file = h5py.File("/home/users/a/algren/scratch/diffusion/shapenet/shapenet.hdf5")
        data= {i:[] for i in ["train", "test", "val"]}
        for i in tqdm(h5_file.keys()):
            for j in data.keys():
                data[j].append(h5_file[i][j][:])
        for j in data.keys():
            _data = np.concatenate(data[j], 0)
            index = np.arange(0, len(_data), 1)
            np.random.shuffle(index)
            data[j] = _data[index, :, :]
        import matplotlib.pyplot as plt
        for i in range(9):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            n = 100

            ax.scatter(data["train"][i, :, 0], data["train"][i, :, 1],
                    data["train"][i, :, 2], marker="o")
    elif False: # create correct pc data size
        PATH = "/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"
        df = pd.read_csv(PATH, dtype=np.float32)
        (sample, mask, idx_number, max_cnstits, X_max, mean, std, n_pc
         ) = preprocess_jets(df)
        new_sample, new_mask= fill_data_in_pc(sample, n_pc, idx_number, max_cnstits)
        pd.DataFrame(new_sample.reshape(new_sample.shape[0], -1)).to_csv(
            PATH.replace(".csv", "_processed.csv")
        )
    else:
        PATH = "/home/users/a/algren/scratch/diffusion/pileup/ttbar_processed.csv"
        pc_jet = PointCloud(PATH)