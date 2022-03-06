
from torch.utils.data.dataset import Dataset
# from pycox.preprocessing.feature_transforms import *
import torch

# def categorical_transformer(X,cat_cols,cont_cols):
#     c = OrderedCategoricalLong()
#     for el in cat_cols:
#         X[:,el] = c.fit_transform(X[:,el])
#     cat_cols = cat_cols
#     if cat_cols:
#         unique_cat_cols = X[:,cat_cols].max(axis=0).tolist()
#         unique_cat_cols = [el + 1 for el in unique_cat_cols]
#     else:
#         unique_cat_cols = []
#     X_cont=X[cont_cols]
#     X_cat=X[cat_cols]
#     return X_cont,X_cat,unique_cat_cols

class general_custom_dataset(Dataset):
    def __init__(self,X,y,x_cat=[]):
        super(general_custom_dataset, self).__init__()
        self.split(X=X,y=y,X_cat=x_cat,mode='train')

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

class chunk_iterator():
    def __init__(self,X,y,cat_X,shuffle,batch_size):
        self.X = X
        self.y = y
        self.cat_X = cat_X
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.valid_cat = not isinstance(self.cat_X, list)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm,:]
            if self.valid_cat: #F
                self.cat_X = self.cat_X[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        if self.valid_cat:  # F
            self.it_cat_X = torch.chunk(self.cat_X,self.chunks)
        else:
            self.it_cat_X = []
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            if self.valid_cat:
                result = (self.it_X[self._index],self.it_cat_X[self._index],self.it_y[self._index])
            else:
                result = (self.it_X[self._index],[],self.it_y[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.train_X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return chunk_iterator(X =self.dataset.X,
                              y = self.dataset.y,
                              cat_X = self.dataset.cat_X,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len