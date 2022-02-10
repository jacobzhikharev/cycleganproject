# -*- coding: utf-8 -*-
"""
Необходимо указать расположение с картинками А и В, расположение весов для каждого из генераторов
и место для сохранения результатов
"""
import argparse
from src.data.dataloaders import paired_dataset_updated_str
from src.model.generator import Generator
import src.utils.config as config

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt

parser = argparse.ArgumentParser(description = 'config generation')
parser.add_argument('-dA','--dir_A', type=str, help='directory for A pictures, default: config.dir_test_A')
parser.add_argument('-dB','--dir_B', type=str, help='directory for B pictures, default: config.dir_test_B ')
parser.add_argument('-drA','--dir_res_A', type=str, help='directory to save transformed A pictures, default: config.dir_resA')
parser.add_argument('-drB','--dir_res_B', type=str, help='directory to save transformed B pictures, default: config.dir_resB')
parser.add_argument('-wG','--wG_dir',type = str,help = 'directory for A weights, default: ./saves/gen_G')
parser.add_argument('-wF','--wF_dir',type = str, help = 'directory for B weights, default: ./saves/gen_F')
args = parser.parse_args()

if(args.dir_res_A):
    drA = args.dir_res_A
else:
    drA = config.dir_resA

if(args.dir_res_B):
    drB = args.dir_res_B
else:
    drB = config.dir_resB

if(args.wF_dir):
    wF = args.wF_dir
else:
    wF = './saves/gen_F'

if(args.wG_dir):
    wG = args.wG_dir
else:
    wG = './saves/gen_G'

if(args.dir_A):
    dA = args.dir_A
else:
    dA = config.dir_test_A

if(args.dir_B):
    dB = args.dir_B
else:
    dB = config.dir_test_B


default_transforms = tt.Compose([tt.Resize(config.image_size),
                     tt.CenterCrop(config.image_size),
                     tt.ToTensor(),
                     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ])

#dataset = paired_dataset(dir_train_A,dir_train_B,transform_train)
dataset = paired_dataset_updated_str(args.dir_A,args.dir_B,default_transforms,requires_flip=False)
dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = False)

Dx = Discriminator().to(config.device)#x and F(y) X == A Y == B
Dy = Discriminator().to(config.device)#y and G(x)
G = Generator().to(config.device)#G:X->Y
F = Generator().to(config.device)#F:Y->X

G.load_state_dict(torch.load(wG,map_location=torch.device(config.device)))
F.load_state_dict(torch.load(wF,map_location=torch.device(config.device)))

iterator = iter(dataloader)
for i in range(dataset.__len__()):
    test = next(iterator)
    labels_gen = G(test[2])
    labels_back = F(test[3])
    torchvision.utils.save_image(labels_gen*0.5+0.5,os.path.join(drA,'res%s'%(test[0][0]))
    torchvision.utils.save_image(labels_back*0.5+0.5,os.path.join(drB,'res%s'%(test[1][0]))
