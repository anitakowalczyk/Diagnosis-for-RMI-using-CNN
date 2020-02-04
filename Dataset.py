from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from torch import FloatTensor
import torch.nn.functional as F
from torch.autograd import Variable

MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


def to_tensor(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * MAX_PIXEL_VAL
    array = (array - MEAN) / STDDEV
    array = np.stack((array,) * 3, axis=1)
    return FloatTensor(array)


class Dataset(Dataset):
    def __init__(self, use_gpu, data_dir, task, part, percent):
        super().__init__()
        self.use_gpu = use_gpu
        self.data_dir = data_dir
        self.task = task

        self.samples = pd.read_csv('{0}-{1}.csv'.format(self.data_dir, task), names=['filename', 'label'])
        self.samples['filename'] = self.samples['filename'].map(lambda i: '0' * (4 - len(str(i))) + str(i))

        if part:
            self.samples = self.samples.sample(frac=percent/100)
            self.samples.index = range(len(self.samples))

        self.paths = ['{0}.npy'.format(filename) for filename in self.samples['filename'].tolist()]
        self.labels = self.samples['label'].tolist()

        negative_weight = np.mean(self.labels)
        self.weights = [negative_weight, 1 - negative_weight]

    def __getitem__(self, index):
        file = self.paths[index]
        axial = np.load(os.path.join(self.data_dir, 'axial', file))
        sagittal = np.load(os.path.join(self.data_dir, 'sagittal', file))
        coronal = np.load(os.path.join(self.data_dir, 'coronal', file))

        axial_tensor = to_tensor(axial)
        sagittal_tensor = to_tensor(sagittal)
        coronal_tensor = to_tensor(coronal)
        label_tensor = FloatTensor([self.labels[index]])

        if self.labels[index] == 1:
            weight = np.array([self.weights[1]])
        else:
            weight = np.array([self.weights[0]])
        weight = FloatTensor(weight)

        return axial_tensor, sagittal_tensor, coronal_tensor, label_tensor, weight

    def __len__(self):
        return len(self.paths)

    def is_in_samples_part(self, filename):
        if filename in self.samples.values:
            return True

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        return F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
