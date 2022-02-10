# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import src.utils.config as config

class R256_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.R = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(256),
        )
    def forward(self,x):
        return x+self.R(x) #Тут сделано как в гитхабе авторов оригинальной статьи т.е. как в resnet
"""
Для избавления от артефактов вместо ConvTransposed используется Upsample+ReflectionPad+Conv2d
В Upsample mode = nearest
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675
https://distill.pub/2016/deconv-checkerboard/
"""
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c7s164 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),
        )
        self.d128 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),
        )
        self.d256 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace = True),
        )
        resnet = []
        for i in range(9):
            resnet.append(R256_block())
        self.R256x9 = nn.Sequential(*resnet)
        self.u128 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace = True),
        )
        self.u64 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace = True),
        )
        self.c7s13 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 3,kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect'),
        )
    def forward(self,x):
        x = self.c7s164(x)
        x = self.d128(x)
        x = self.d256(x)
        x = self.R256x9(x)
        x = self.u128(x)
        x = self.u64(x)
        x = self.c7s13(x)
        return torch.tanh(x)

def Loss_Gen(G,F,Dx,Dy,real_image_A,real_image_B,fake_image_A,fake_image_B,gan_loss,cycle_loss):#,id_loss):
    fake_preds_A = Dx(fake_image_A)
    fake_preds_B = Dy(fake_image_B)
    L_gan_F = gan_loss(fake_preds_A, torch.ones_like(fake_preds_A))
    L_gan_G = gan_loss(fake_preds_B, torch.ones_like(fake_preds_B))
    recon_A = F(fake_image_B)
    recon_B = G(fake_image_A)
    L_cyc_G = cycle_loss(real_image_A, recon_A)
    L_cyc_F = cycle_loss(real_image_B, recon_B)
    return L_gan_G+L_gan_F+(L_cyc_G+L_cyc_F)*config.lam
