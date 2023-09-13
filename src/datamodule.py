"General datamodule for pytorch"
import torch as T
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
from itertools import chain, cycle
from tqdm import tqdm
import numpy as np

# internal

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MultiFileDataset(IterableDataset):
    def __init__(self, data_lst:list, batch_size:int, processing_func=None):
        self.data_lst=data_lst
        self.batch_size=batch_size
        
        if (processing_func is not None):
            self._process_data = processing_func
    
    @property
    def shuffled_data_list(self):
        return random.sample(self.data_lst, len(self.data_lst))

    def _process_data(self,*args):
        raise NotImplementedError

    def __iter__(self):
        return iter(self._process_data(self.shuffled_data_list))
    
    
    @classmethod
    def split_datasets(cls, data_list, batch_size, max_workers, processing_func=None):
        "split paths into list pr worker"
        
        split_data_lst = list(chunks(data_list,
                                     int(np.ceil(len(data_list)/max_workers))))
        
        if max_workers < len(split_data_lst):
            raise ValueError("max_workers higher than number of fils")
        
        return [cls(data_lst=path, batch_size=batch_size, processing_func=processing_func)
                for path in split_data_lst]
        
class MultiStreamDataLoader:
    def __init__(self, datasets, data_kw=None):
        self.data_kw=data_kw
        if self.data_kw is None:
            self.data_kw={}
        self.datasets = datasets
        self.loaders = self.get_stream_loaders()
        
    def get_stream_loaders(self):
        return zip(*[DataLoader(dataset, **self.data_kw)
                     for dataset in self.datasets])
        
    def __iter__(self):
        for batch_parts in self.get_stream_loaders():
            # data = batch_parts
            for i in batch_parts:
                yield i

# class JetPhysics(MultiFileDataset):
#     def _process_data(self, paths_lst):
#         for path in paths_lst:
#             # worker = T.utils.data.get_worker_info()
#             # worker_id = id(self) if worker is not None else -1
#             print(path)
#             x = load_csv([path], verbose=False)
#             for i in batch(x[1], self.batch_size):
#                 yield {"mask": i} #, worker_id

if __name__ == "__main__":
    import hydra
    from tools import misc
    from physics import load_csv
    
    
    paths = misc.load_yaml("/srv/beegfs/scratch/groups/rodem/datasets/pileup_jets/top/path_lists.yaml")
    
    datasets = JetPhysics.split_datasets(paths["train_path"][:4], batch_size=1024,
                                                     max_workers=4)
    loader = MultiStreamDataLoader(datasets)
    data = []
    for i in tqdm(loader):
        data.append(i)
        # print(i)
        # break
        pass
