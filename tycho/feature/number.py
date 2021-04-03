import torch
import numpy as np


class NumericNormalizer():
    def __init__(self):
        self.abs_mean = None

    def fit(self, data):
        #self.abs_mean = np.mean(np.abs(data))
        self.abs_mean = 1
        
    def __call__(self, data):
        featurize_data = torch.zeros(len(data),1)
        for i, point in enumerate(data):
            featurize_data[i][0] = point/self.abs_mean
        return featurize_data

    def reverse(self, tensor):
        data = []
        for ele in tensor:
            real = ele[0] * self.abs_mean
            data.append(real)
        return data
