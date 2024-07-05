import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from sklearn.model_selection import train_test_split

def create_dataset(args):


    if args.dataset == 'cifar10':

        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2470, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

        train_dataset = CIFAR10(root='/home/data/cifar10', train=True, download=False, transform=train_transform)
        test_dataset = CIFAR10(root='/home/data/cifar10', train=False, download=False, transform=test_transform)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

        

    return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers    

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
