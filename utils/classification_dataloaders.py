import os.path

import numpy as np
from torchvision.datasets import MNIST,FashionMNIST,CIFAR10,SVHN
from torchvision.transforms import ToTensor,Pad,Compose
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from ffcv.fields import IntField,RGBImageField,NDArrayField
import ffcv.transforms as transforms
from ffcv.writer import DatasetWriter
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from typing import List
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
def get_dataloaders_FFCV(dataset_string,batch_size,val_factor):
    if dataset_string=='CIFAR10':
        dataset=CIFAR10( root="../data",
        train=True,
        download=True,
        # transform=ToTensor()
                         )

        test_set = CIFAR10(root="../data",
                          train=False,
                          download=True,
                          # transform=ToTensor()
                           )


    elif dataset_string=='SVHN':
        dataset=SVHN( root="../data",
        split='train',
        download=True,
        transform=ToTensor())
        test_set = SVHN(root="../data",
                          split='test',
                          download=True,
                          # transform=ToTensor()
                        )


    elif dataset_string=='FashionMNIST':
        dataset=FashionMNIST( root="../data",
        train=True,
        download=True,
        # transform=Compose([Pad(2),ToTensor()])
                              )
        dataset.data = dataset.data.unsqueeze(1)
        test_set=FashionMNIST( root="../data",
        train=False,
        download=True,
        # transform=Compose([Pad(2),ToTensor()])
                               )
        test_set.data = test_set.data.unsqueeze(1)

    elif dataset_string=='MNIST':
        dataset=MNIST( root="../data",
        train=True,
        download=True,
        # transform=Compose([Pad(2),ToTensor()])
                       )
        dataset.data = dataset.data.unsqueeze(1)

        test_set=MNIST( root="../data",
        train=False,
        download=True,
        # transform=Compose([Pad(2),ToTensor()])
                        )
        test_set.data = test_set.data.unsqueeze(1)

    og_len = len(dataset)
    val_size =int(round(val_factor*og_len))
    split_sizes=[og_len-val_size,val_size]
    train_set, val_set = torch.utils.data.random_split(dataset, split_sizes)
    datasets = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    if not os.path.exists('../ffcv_datasets_tmp'):
        os.makedirs('../ffcv_datasets_tmp')
    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'../ffcv_datasets_tmp/{dataset_string}_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)
    loaders = {}
    for name in ['train','val', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), transforms.ToTensor(), transforms.ToDevice('cuda:0'), transforms.Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        # if dataset_string in ['MNIST','FashionMNIST']:
        #     image_pipeline.extend([
        #         torchvision.transforms.Pad(2)
        #     ])

        image_pipeline.extend([
            transforms.ToTensor(),
            transforms.ToDevice('cuda:0', non_blocking=True),
            transforms.ToTorchImage(),
            transforms.Convert(torch.float32),

        ])
        if dataset_string=='CIFAR10':
            image_pipeline.extend([
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        else:
            image_pipeline.extend([
                torchvision.transforms.Normalize(0.5, 0.5)
            ])
        # Create loaders
        loaders[name] = Loader(f'../ffcv_datasets_tmp/{dataset_string}_{name}.beton',
                               batch_size=batch_size,
                               num_workers=1,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})

    # train_loader = DataLoader(train_set,batch_size=batch_size)
    # val_loader = DataLoader(val_set,batch_size=batch_size)
    # test_loader = DataLoader(test_set,batch_size=batch_size)
    return loaders['train'],loaders['val'],loaders['test']



def get_dataloaders(dataset_string,batch_size,val_factor):
    if dataset_string=='CIFAR10':
        dataset=CIFAR10( root="../data",
        train=True,
        download=True,
        transform=ToTensor())
        test_set = CIFAR10(root="../data",
                          train=False,
                          download=True,
                          transform=ToTensor())
    elif dataset_string=='SVHN':
        dataset=SVHN( root="../data",
        split='train',
        download=True,
        transform=ToTensor())
        test_set = SVHN(root="../data",
                          split='test',
                          download=True,
                          transform=ToTensor())


    elif dataset_string=='FashionMNIST':
        dataset=FashionMNIST( root="../data",
        train=True,
        download=True,
        transform=Compose([Pad(2),ToTensor()]))
        test_set=FashionMNIST( root="../data",
        train=False,
        download=True,
        transform=Compose([Pad(2),ToTensor()]))
    elif dataset_string=='MNIST':
        dataset=MNIST( root="../data",
        train=True,
        download=True,
        transform=Compose([Pad(2),ToTensor()]))
        test_set=MNIST( root="../data",
        train=False,
        download=True,
        transform=Compose([Pad(2),ToTensor()]))

    og_len = len(dataset)
    val_size =int(round(val_factor*og_len))
    split_sizes=[og_len-val_size,val_size]
    train_set, val_set = torch.utils.data.random_split(dataset, split_sizes)
    train_loader = DataLoader(train_set,batch_size=batch_size)
    val_loader = DataLoader(val_set,batch_size=batch_size)
    test_loader = DataLoader(test_set,batch_size=batch_size)
    return train_loader,val_loader,test_loader

# if __name__ == '__main__':
#     dataset=FashionMNIST( root="../data",
#     train=True,
#     download=True,
#     transform=Compose([Pad(2),ToTensor()]),
#                    )
#     og_len = len(dataset)
#     val_factor=0.05
#     val_size =int(round(val_factor*og_len))
#     split_sizes=[og_len-val_size,val_size]
#
#     train_set, val_set = torch.utils.data.random_split(dataset, split_sizes)
#     print(len(train_set))
#     train_loader = DataLoader(train_set,batch_size=1)
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print(data.shape)
#         print(target)
#
