# -*- coding: utf-8 -*-
from src.utils.other import buffer, pr_to_plot, func2, weights_init
from src.data.dataloaders import paired_dataset,paired_dataset_updated
from src.model.generator import Generator,Loss_Gen
from src.model.discriminator import Discriminator,Loss_Disc
from src.utils import config

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

parser = argparse.ArgumentParser(description = 'configure training')
parser.add_argument('-n','--n_epochs',type = int,help = 'number of epochs, default: 200')
parser.add_argument('-sd','--save_dir',type = str,help = 'directory for saves, default: saves')
parser.add_argument('-dA','--dir_A', type=str, help='directory for A pictures, default: config.dir_train_A')
parser.add_argument('-dB','--dir_B', type=str, help='directory for B pictures, default: config.dir_train_B ')
args = parser.parse_args()

if(args.dir_A):
    dA = args.dir_A
else:
    dA = config.dir_train_A

if(args.dir_B):
    dB = args.dir_B
else:
    dB = config.dir_train_B
if(args.n_epochs):
    n_epochs = args.n_epochs
else:
    n_epochs = args.n_epochs
if(args.save_dir):
    dsave = args.save_dir
else:
    dsave = config.dir_saves

# TODO: Change to correct model

default_transforms = tt.Compose([tt.Resize(config.image_size),
                     tt.CenterCrop(config.image_size),
                     tt.ToTensor(),
                     tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ])


#dataset = paired_dataset(dir_train_A,dir_train_B,transform_train)
dataset = paired_dataset_updated(dA,dB, default_transforms,requires_flip=True)
dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = True)

class d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            nn.Conv2d(3,3,3,1,1),
            torch.nn.Sigmoid(),
        )
    def forward(self,x):
        return self.s1(x)


class g(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = torch.nn.Sequential(
        torch.nn.ReLU(),
        nn.Conv2d(3,3,3,1,1),
        torch.nn.Sigmoid())
    def forward(self,x):
        return self.s1(x)

gan_loss = MSELoss()
cycle_loss = L1Loss()
buffer_a = buffer()
buffer_b = buffer()

# Dx = Discriminator().to(device)#x and F(y) X == A Y == B
# Dy = Discriminator().to(device)#y and G(x)
# G = Generator().to(device)#G:X->Y
# F = Generator().to(device)#F:Y->X
Dx = d().to(config.device)#x and F(y) X == A Y == B
Dy = d().to(config.device)#y and G(x)
G = g().to(config.device)#F:X->Y
F = g().to(config.device)#F:Y->X

Dx.apply(weights_init)
Dy.apply(weights_init)
G.apply(weights_init)
F.apply(weights_init)

optimizer_D = torch.optim.Adam(chain(Dx.parameters(),Dy.parameters()),lr=config.lr,betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(chain(G.parameters(),F.parameters()),lr=config.lr,betas=(0.5, 0.999))

"""
Ипользуется именно такой вид т.к. для иначе придется либо иметь отдельный лосс для кажого случая,
либо следить за сохранением графов вычислений
"""

sch_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,func2,last_epoch=-1)
sch_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,func2,last_epoch=-1)

'''
Лоссы состоят из сумм лоссов генераторов и дискриминаторов
Я не делаю перевод модели в train/eval, т.к. это не влияет на их поведение(и на InstanceNorm в данном случае тоже)
Изображения уже загружены на gpu paired_dataset_updated(), это немного ускоряет обучение
Если данные не помещаются в память, то необходимо использовать paired_dataset()
'''

Generators_loss = []
Discriminators_loss = []
for epoch in range(n_epochs):
    Generators_loss_per_epoch = []
    Discriminators_loss_per_epoch = []
    for img_A,img_B in tqdm(dataloader):
        #img_A = img_A.to(device)
        #img_B = img_B.to(device)
        fake_A = F(img_B)
        fake_B = G(img_A)
        Loss_Dxy = 1.0/2.0*(Loss_Disc(Dx,img_A,fake_A,gan_loss,buffer_a)+
                        Loss_Disc(Dy,img_B,fake_B,gan_loss,buffer_b))
        optimizer_D.zero_grad()
        Loss_Dxy.backward()
        optimizer_D.step()

        Loss_GF = Loss_Gen(G,F,Dx,Dy,img_A,img_B,fake_A,fake_B,gan_loss,cycle_loss)
        optimizer_G.zero_grad()
        Loss_GF.backward()
        optimizer_G.step()
        Discriminators_loss_per_epoch.append(Loss_Dxy.detach().cpu())
        Generators_loss_per_epoch.append(Loss_GF.detach().cpu())
        #img_A = img_A.detach().cpu()
        #img_B = img_B.detach().cpu()
        torch.cuda.empty_cache()
    sch_G.step()
    sch_D.step()
    if epoch%10 == 0:
        torch.save(G.state_dict(),os.path.join(dsave,'gen_G'))
        torch.save(F.state_dict(),os.path.join(dsave,'gen_F'))
        torch.save(Dx.state_dict(),os.path.join(dsave,'disc_Dx'))
        torch.save(Dy.state_dict(),os.path.join(dsave,'disc_Dy'))
        torch.save(optimizer_D.state_dict(),os.path.join(dsave,'opt_D'))
        torch.save(optimizer_G.state_dict(),os.path.join(dsave,'opt_G'))
    Discriminators_loss.append(np.mean(Discriminators_loss_per_epoch))
    Generators_loss.append(np.mean(Generators_loss_per_epoch))
    print("Generators_loss = %f, Discriminators_loss = %f,epoch = %i, learning_rate = %f"
          %(Generators_loss[-1],Discriminators_loss[-1],epoch+1,sch_D.get_last_lr()[0]))

plt.figure(figsize=(10, 8))
plt.title('Discriminators loss')
plt.plot(Discriminators_loss)

plt.figure(figsize=(10, 8))
plt.title('Generators loss')
plt.plot(Generators_loss)
plt.show()
with open(os.path.join(dsave,'disc_loss'), "wb") as fp:
        pickle.dump(Discriminators_loss, fp)
with open(os.path.join(dsave,'gen_loss'), "wb") as fp:
        pickle.dump(Generators_loss, fp)
