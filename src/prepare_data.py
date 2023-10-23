"function to setup data correct for dataloader"
from tqdm import tqdm
import numpy as np
import h5py

def prepare_mnist(df, dummy_val=-1, max_cnstits=None):
    "dummy 3d mnist dataset"
    X = df[df.columns[1:]].to_numpy()
    y = df[df.columns[0]].to_numpy()
    X = X[np.isin(y, [7])]
    X = X.reshape(X.shape[0], -1, 3)
    # X = np.moveaxis(X, 1,2) # making it pytorch compatible (N, C, ...)

    # normalizing the data
    mask = X!=-1
    if max_cnstits is None:
        max_cnstits = mask.sum(1).max()
    min_cnstits = mask.sum(1).min()
    X = X[:,:max_cnstits]
    mask = mask[:,:max_cnstits]
    X_max = np.max(X, (0,2))[None].T[None]

    # X = X/X_max
    X[~mask] = -1

    mean = np.array([X[:,:,i][X[:,:,i]!= dummy_val].mean()
                     for i in range(X.shape[2])])[None, :]
    
    std = np.array([X[:,:,i][X[:,:,i]!= dummy_val].std()
                    for i in range(X.shape[2])])[None, :]

    return np.float32(X), np.all(mask, -1), min_cnstits, max_cnstits, mean, std, len(X)

def preprocess_jets(df):
    "pileup jet datasets"
    jet_numbers, n_cnstits = np.unique(df.jetnumber, return_counts=True)
    max_cnstits = np.max(n_cnstits)
    
    X = df[["px", "py","pz"]].values.T

    X_max = np.max(X, 1)[:, None]
    
    
    return (X, n_cnstits, df.jetnumber.values, max_cnstits, X_max,
            X.mean(1)[:, None], X.std(1)[:, None], len(jet_numbers))


def prepare_shapenet(h5_file):
    _data= {i:[] for i in ["train", "test", "val"]}
    for i in tqdm(h5_file.keys()):
        for j in _data.keys():
            _data[j].append(h5_file[i][j][:])
    data = []
    for j in _data.keys():
        __data = np.concatenate(_data[j], 0)
        index = np.arange(0, len(__data), 1)
        np.random.shuffle(index)
        data.append(__data[index, :, :])
        
    data = np.concatenate(data, 0)
    mask = np.ones(data.shape[:-1])==1 ## all are True
    data = np.swapaxes(data, 1,2)
    std = data.std((0,2))[:, None]
    mean = data.mean((0,2))[:, None]
    raise TypeError("data shape should be changed and mean/std")
    return np.float32(data), mask, data.shape[-1], mean, std, len(data)

def matrix_to_point_cloud(sample, idx_numbers, num_per_event_max=None):
    "sample: n_features x pc"
    n_pc, num_per_event = np.unique(idx_numbers, return_counts=True)
    
    #
    if num_per_event_max is None:
        # define the max number of cnts
        num_per_event_max= num_per_event.max()
    elif num_per_event_max is not None:
        # use predefined number of conts
        mask_max_cnts = num_per_event< num_per_event_max
        mask_sample = np.isin(idx_numbers,  n_pc[mask_max_cnts])
        sample = sample[np.ravel(mask_sample)]
        num_per_event = num_per_event[mask_max_cnts]
        n_pc = n_pc[mask_max_cnts]
    n_events = len(n_pc)

    # create mask tensor
    mask = np.arange(0, num_per_event_max)
    mask = np.expand_dims(mask, 0)
    mask = mask < np.expand_dims(num_per_event,1)

    # create tensor with padded elements
    padded_tens = np.ones((n_events, num_per_event_max, sample.shape[1]))*-1

    # add original sample to padded elements
    padded_tens[mask] = sample

    return padded_tens, mask

def fill_data_in_pc(sample, idx_numbers):
    # Deprecated!!!!!
    "sample: n_features x pc"
    n_pc, max_cnstits = np.unique(idx_numbers, return_counts=True)

    n_pc = len(n_pc)
    max_cnstits = np.max(max_cnstits)
    
    new_sample = np.ones((n_pc, max_cnstits, sample.shape[1]))*-999
    new_mask = np.zeros((n_pc, max_cnstits))
    for nr, idx in tqdm(enumerate(np.unique(idx_numbers)), total=n_pc):
        _sample=sample[idx_numbers==idx]
        new_sample[nr, :_sample.shape[0], :_sample.shape[1]] = _sample
        new_mask[nr, :_sample.shape[0]] = 1
    
    return new_sample, new_mask

if __name__ == "__main__":
    PATH = "/home/users/a/algren/scratch/diffusion/pileup/ttbar.csv"
    import pandas as pd
    df = pd.read_csv(PATH, dtype=np.float32)
    df=df.rename(columns={i:i.replace(" ", "") for i in df.columns})
    df["phi"] = np.arctan(df["py"]/df["px"])
    df["pT"] = df["px"]/np.cos(df["phi"])
    df["eta"] = np.arcsinh(df["pz"]/df["pT"]) 



    # (sample, mask, idx_number, max_cnstits, X_max, mean, std, n_pc
    #     ) = preprocess_jets(df)
    # new_sample, new_mask= preprocess_jets(sample, n_pc, idx_number, max_cnstits)
    # pd.DataFrame(new_sample.reshape(new_sample.shape[0], -1)).to_csv(
    #     PATH.replace(".csv", "_processed.csv")
    # )