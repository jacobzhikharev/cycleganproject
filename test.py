# -*- coding: utf-8 -*-
from src.utils.other import buffer, pr_to_plot, func2, weights_init
from src.data.dataloaders import paired_dataset,paired_dataset_updated
from src.model.generator import Generator,Loss_Gen
from src.model.discriminator import Discriminator,Loss_Disc
import src.utils.config as config

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm.notebook import tqdm
from torch.nn import L1Loss, MSELoss
from itertools import chain
import pickle

parser = argparse.ArgumentParser(description = 'config generation')
# parser.add_argument('dir_A', type=str, help='directory for A pictures')
# parser.add_argument('dir_B', type=str, help='directory for B pictures')
# parser.add_argument('dir_res_A', type=str, help='directory to save transformed A pictures')
# parser.add_argument('dir_res_B', type=str, help='directory to save transformed B pictures')
# parser.add_argument('model_type',type = str,help = 'model type: w2p, h2z, f2l')
parser.add_argument('-w','--weights',type = str, help = 'directory with weights, default = ./weights')
args = parser.parse_args()

# print(args.dir_A)
# print(args.dir_B)
# print(args.dir_res_A)
# print(args.dir_res_B)
# print(args.dir_A)
# print(args.model_type)
print(args.weights)

if(args.weights):
    w_dir = args.weights
else:
    w_dir = './weights'

print(w_dir)
#image_size = 256

# default_transforms = tt.Compose([tt.Resize(config.image_size),
#                      tt.CenterCrop(config.image_size),
#                      tt.ToTensor(),
#                      tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                      ])
#
# ds = paired_dataset(config.dir_train_A,config.dir_train_B,default_transforms)
# dl = DataLoader(ds,batch_size = 1,shuffle = True)
#
# iterator_train = iter(dl)
#
# test = next(iterator_train)
# plt.figure()#figsize=(16, 8))
# plt.subplot(1, 4, 1)
# plt.title('real image')
# plt.imshow(pr_to_plot(test[0][0].detach().cpu()))
# plt.axis('off')
# plt.subplot(1, 4, 2)
# plt.title('real label')
# plt.imshow(pr_to_plot(test[1][0].detach().cpu()))
# plt.axis('off')
# plt.show()

#Good
# Dx = Discriminator().to(config.device)#x and F(y) X == A Y == B
# Dy = Discriminator().to(config.device)#y and G(x)
# G = Generator().to(config.device)#G:X->Y
# F = Generator().to(config.device)#F:Y->X
# #Good
# Dx.apply(weights_init)
# Dy.apply(weights_init)
# G.apply(weights_init)
# F.apply(weights_init)
# #Good
# G.load_state_dict(torch.load('./weights/w2p_final_gen_G',map_location=torch.device(config.device)))
# F.load_state_dict(torch.load('./weights/w2p_final_gen_F',map_location=torch.device(config.device)))
# #Good
# print(G)
# print(F)
# print(Dy)
# print(Dx)
#plt.show()
print("OK")
