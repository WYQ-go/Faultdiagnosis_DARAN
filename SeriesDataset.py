import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from Config import DataConfig


def onehot_encode(target):
    label = np.unique(target).tolist()
    # if 0 in label:
    #     label.remove(0)
    onehot_vector = np.zeros((len(target), DataConfig.class_num))
    for i in label:
        onehot_vector[np.where(target == i), int(i)] = 1
    return onehot_vector


class FDDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode
        data = pd.read_csv(path, encoding='utf-8', low_memory=False).to_numpy()
        self.data = torch.FloatTensor(data[:, 1:])
        target = onehot_encode(data[:, 0])
        self.target = torch.FloatTensor(target)
        self.label = data[:, 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].reshape(1, 800), self.target[index]


def prepDataloader(path, mode, batch_size, n_jobs=0):
    dataset = FDDataset(path)
    return DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=n_jobs, pin_memory=True)


class TLTrainDataLoader:
    def __init__(self, SPath, TTrPath, batch_size):
        self.batch_size = batch_size
        self.Sdata = pd.read_csv(SPath, encoding='utf-8', low_memory=False).to_numpy()
        TTrData = pd.read_csv(TTrPath, encoding='utf-8', low_memory=False).to_numpy()
        self.TTrData = np.empty([0, TTrData.shape[1]])
        for label in DataConfig.share_labels:
            self.TTrData = np.concatenate((self.TTrData, TTrData[np.where(TTrData[:, 0] == label), :][0]), axis=0)
        self.step = (len(self.Sdata) + len(self.TTrData)) // self.batch_size
        if (len(self.Sdata) + len(self.TTrData)) % (self.batch_size) != 0:
            self.step += 1
        self.batch_ratio = len(self.Sdata) / (len(self.Sdata) + len(self.TTrData))

    def __len__(self):
        return self.step

    def __iter__(self):
        sidx = np.arange(len(self.Sdata))
        tidx = np.arange(len(self.TTrData))
        np.random.shuffle(sidx)
        np.random.shuffle(tidx)
        source_batch = int(self.batch_size * self.batch_ratio)
        target_batch = self.batch_size - source_batch
        for step in range(self.step):
            sindex = sidx[step * source_batch:(step + 1) * source_batch]
            tindex = tidx[step * target_batch:(step + 1) * target_batch]
            source_data = self.Sdata[sindex, 1:]
            source_target = self.Sdata[sindex, 0]
            target_data = self.TTrData[tindex, 1:]
            yield torch.FloatTensor(source_data.reshape(len(source_data), 1, DataConfig.sampleLen)), \
                  torch.FloatTensor(onehot_encode(source_target)), \
                  torch.FloatTensor(target_data.reshape(len(target_data), 1, DataConfig.sampleLen))


if __name__ == '__main__':
    root = "D:/Projects/partial_domain_adaption/CWRU-master"
    path = root + '/' + 'Series/CWRU_10/'
    train_path = path + "/" + "source_train.csv"
    tr_set = FDDataset(train_path)

    # tr_set = TLTrainDataLoader(DataConfig.Tfile_path, DataConfig.DTRfile_path, 1024, mode='unsupersivsed')
    for x, y in tr_set:
        a = x
        b = y
    pass
