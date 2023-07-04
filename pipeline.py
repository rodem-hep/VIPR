import sys
from copy import deepcopy
from functools import partial
from typing import Mapping

import torch as T
from torch.utils.data import Dataset,DataLoader
from tools import misc
import hydra

class ImageEnhancement(Dataset):
    def __init__(self, dataloader, img_enhancement=None):
        self.dataloader=dataloader
        self.img_enhancement=img_enhancement

    def __iter__(self):
        for i, _ in self.dataloader:
            if self.img_enhancement is not None:
                cvx = i.clone()
                for trans in self.img_enhancement:
                    cvx = trans(cvx)
                yield i, cvx
            else:
                yield i, None

class ImageModule():
    def __init__(self, *,
                 train_set: partial, test_set: partial,
                 loader_config: Mapping, img_enc:partial=None
                 ):
        self.img_enc=img_enc
        self.loader_config = loader_config
        self.train_sample=train_set()
        self.test_sample=test_set()

        if img_enc is not None:
            self._img_enc = self.img_enc

    def _img_enc(self, *args):
        return args

    def train_dataloader(self) -> DataLoader:
        return self._img_enc(DataLoader(self.train_sample,
                                       **self.loader_config))

    def test_dataloader(self, run_img_enc=False) -> DataLoader:
        test_config = deepcopy(self.loader_config)
        test_config["drop_last"] = False
        test_config["shuffle"] = False
        if run_img_enc:
            return self._img_enc(DataLoader(self.test_sample,
                                        **test_config))
        else:
            return DataLoader(self.test_sample, **test_config)
        

if __name__ == "__main__":
    # %matplotlib widget
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
        
