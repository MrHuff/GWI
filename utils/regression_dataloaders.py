from utils.dataloaders import *
import numpy as np
import torch
import logging
import os
from os import path
from sklearn.model_selection import KFold
import pandas as pd
import zipfile
import urllib.request
from sklearn.model_selection import train_test_split

class general_custom_dataset_regression(Dataset):
    def __init__(self,X_tr,y_tr,X_val,y_val,X_test,y_test):
        super(general_custom_dataset_regression, self).__init__()
        self.split(X=X_tr,y=y_tr,X_cat=[],mode='train')
        self.split(X=X_val,y=y_val,X_cat=[],mode='val')
        self.split(X=X_test,y=y_test,X_cat=[],mode='test')

    def split(self,X,y,mode='train',X_cat=[]):
        setattr(self,f'{mode}_y', y.float() if torch.is_tensor(y) else torch.from_numpy(y).float())
        setattr(self, f'{mode}_X', X.float() if torch.is_tensor(X) else torch.from_numpy(X).float())
        self.cat_cols = False
        if not isinstance(X_cat,list):
            self.cat_cols = True
            setattr(self, f'{mode}_cat_X',X_cat.long() if torch.is_tensor(X_cat) else torch.from_numpy(X_cat.astype('int64').values).long())

    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        if self.cat_cols:
            self.cat_X = getattr(self,f'{mode}_cat_X')
        else:
            self.cat_X = []

    def __getitem__(self, index):
        if self.cat_cols:
            return self.X[index,:],self.cat_X[index,:],self.y[index]
        else:
            return self.X[index,:],self.cat_X,self.y[index]

    def __len__(self):
        return self.X.shape[0]

class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
            "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
            "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
            "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            'boston': "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            'naval':"https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
            'KIN8NM':"https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
            'protein':"https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"

        }
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path + "UCI"):
            os.mkdir(self.data_path + "UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        # uncomment for issues with urllib
        if not path.exists(self.data_path + "UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path + "UCI/" + file_name)
        data = None

        if self.name == "boston":
            data = pd.read_csv(self.data_path + 'UCI/housing.data',
                               header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path + 'UCI/Concrete_Data.xls',
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "energy":
            data = pd.read_excel(self.data_path + 'UCI/ENB2012_data.xlsx',
                                 header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "power":
            zipfile.ZipFile(self.data_path + "UCI/CCPP.zip").extractall(self.data_path + "UCI/CCPP/")
            data = pd.read_excel(self.data_path + 'UCI/CCPP/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data
        elif self.name == "protein":
            data = pd.read_csv(self.data_path + 'UCI/CASP.csv',
                               header=0, delimiter=',').iloc[:, ::-1]
            data = data.values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=0, delimiter=';')
            data = data.values

            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+')
            data = data.values

            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "naval":
            zipfile.ZipFile(self.data_path + "UCI/CBM_Dataset.zip").extractall(self.data_path + "UCI/UCI CBM Dataset/")

            data = pd.read_csv(self.data_path + 'UCI/UCI CBM Dataset/UCI CBM Dataset/data.txt',
                               header=0, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "KIN8NM":
            data = pd.read_csv(self.data_path + 'UCI/dataset_2175_kin8nm.csv',
                               header=0)
            data = data.values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        self.data = self.data.astype('float')
        kf = KFold(n_splits=self.n_splits)
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1
        self.data_splits = kf.split(data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

    def get_split(self, split=-1):

        if split == -1:
            split = 0

        if 0<=split and split<=self.n_splits:
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            self.empirical_sigma = y_stds
            x_stds[x_stds==0]=1.0
            y_stds[y_stds==0]=1.0
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds

            X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.11)

            inps = torch.from_numpy(X_train).float()
            tgts = torch.from_numpy(y_train).float()

            inps_val = torch.from_numpy(X_val).float()
            tgts_val = torch.from_numpy(y_val).float()

            inps_test = torch.from_numpy(x_test).float()
            tgts_test = torch.from_numpy(y_test).float()
            return inps,tgts,inps_val,tgts_val,inps_test,tgts_test

    def get_split_version_2(self, split=-1):

        if split == -1:
            split = 0

        if 0<=split and split<=self.n_splits:
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index,
                                    :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0)**0.5
            x_stds[x_stds==0]=1.0
            self.empirical_sigma = y_stds
            x_train = (x_train - x_means)/x_stds
            y_train = (y_train - y_means)/y_stds
            x_test = (x_test - x_means)/x_stds
            y_test = (y_test - y_means)/y_stds

            # X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.11, random_state = 42)

            inps = torch.from_numpy(x_train).float()
            tgts = torch.from_numpy(y_train).float()

            inps_val = torch.from_numpy(x_test).float()
            tgts_val = torch.from_numpy(y_test).float()

            inps_test = torch.from_numpy(x_test).float()
            tgts_test = torch.from_numpy(y_test).float()
            return inps,tgts,inps_val,tgts_val,inps_test,tgts_test

def get_regression_dataloader(dataset,fold,bs):
    ds = UCIDatasets(name=dataset,data_path='local_UCI_storage',n_splits=10)
    x_tr,y_tr,x_val,y_val,x_tst,y_tst = ds.get_split(fold)
    dataset = general_custom_dataset_regression(x_tr,y_tr,x_val,y_val,x_tst,y_tst)
    return custom_dataloader(dataset=dataset,batch_size=bs,shuffle=True),ds