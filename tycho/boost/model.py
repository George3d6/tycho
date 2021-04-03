import numpy as np
import torch
from torch.utils.data import TensorDataset

from tycho.util.log import log
from tycho.util.fold import gen_folds
from tycho.feature.detect import get_featurizers_class, detect_df_types
from tycho.boost.nets import BoostNet


class Model():
    def __init__(self, featurizer_dict, target, target_type):
        self.featurizer_dict = featurizer_dict
        self.target = target
        self.model = None
        self.target_type = target_type

    def featurize(self, df):
        featurized_columns = {}
        for col, featurizer in self.featurizer_dict.items():
            featurized_columns[col] = featurizer(df[col])

        X = None
        Y = None
        for col, data in featurized_columns.items():
            if col != self.target:
                if X is None:
                    X = data
                else:
                    X = torch.cat([X,data],1)
            else:
                Y = data

        return X, Y

    def reverse(self, Y):
        return self.featurizer_dict[self.target].reverse(Y)

    def fit(self, df):
        X, Y = self.featurize(df)

        folds = gen_folds((X,Y),5)
        folds = [TensorDataset(x,y) for x,y in folds]

        self.model = BoostNet(self.target_type)
        self.model.fit(folds)

    def infer(self, df):
        log.debug('Making an inference')
        X, _ = self.featurize(df)
        Yp = self.model.infer(X)
        return self.reverse(Yp)

def fit(df, target):
    log.debug('Fitting a model')

    log.info(f'Shuffling data')
    df = df.sample(frac=1, random_state=len(df)).reset_index(drop=True)

    df_types = detect_df_types(df)
    featurizer_class_dict = get_featurizers_class(df_types)
    log.info(f'Using featurizers: {featurizer_class_dict}')

    log.debug('Fitting featurizers')
    featurizer_dict = {}
    for col, featurizer_class in featurizer_class_dict.items():
        featurizer_dict[col] = featurizer_class()
        featurizer_dict[col].fit(list(df[col]))
    log.debug('Fitting featurizers')

    log.debug('Preparing encoder')

    log.debug('Prepared encoder')

    model = Model(featurizer_dict, target, df_types[target])
    model.fit(df)
    return model
