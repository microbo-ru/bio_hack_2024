import argparse
import json
import os
import shutil
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import zipfile

parser = argparse.ArgumentParser(prog='example')
parser.add_argument('-i', '--input', required=True, help='Input file')
parser.add_argument('-m', '--model', required=True, help='Output file')

if __name__ == '__main__':
    # # envzy explorer can't find imports outside global namespace
    # import pandas as pd
    # # from data import arr

    args = parser.parse_args()

    # print(args)

    # df = pd.read_csv(f'{args.input}/HandInfo.csv')

    # df.to_csv(f'{args.model}', sep='\t')
    # # with open(args.model, 'w') as f:
    # #     f.write(str(float(pd.Series(arr()).mean())))

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define parameters
    # dataroot = "/home/vladimir/Work/microbo/hands_prep/Hands/Hands/"
    dataroot = f'{args.input}/hands_prep/Hands/Hands/'
    nb_channels = 3 # For RGB images, but if you use grayscale images, ToTensor() will replicate the single channel into three channels, so you should not have to modify anything
    image_resize = 64

    batch_size = 128
    nb_gpu = 1
    nb_workers = 4 # based on system resources

    # GPU or CPU (Not having at least 1 GPU can prevent code from working)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and nb_gpu > 0) else "cpu")

    # Create the dataset by applying transformation to our images
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_resize),
                                transforms.CenterCrop(image_resize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    
    print(f'Number of downloaded images: {len(dataset)}')

    with open(args.model, 'w') as f:
      f.write(f'test demo: {len(dataset)}')
