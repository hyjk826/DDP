import random 
import os 
import numpy as np 
import torch 
import json 
from tqdm import tqdm 
import sys 

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Logger(object):
    def __init__(self, no_save=False):
        self.terminal = sys.stdout
        self.file = None
        self.no_save = no_save
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if not self.no_save: 
            self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1 and not self.no_save:
            self.file.write(msg)
            self.file.flush()
    def flush(self): 
        pass

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{:25s}: {}\n'.format(k, v))
        else:
            print('{:25s}: {}'.format(k, v))

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

@torch.no_grad()
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc='Test...', leave=False) :
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total