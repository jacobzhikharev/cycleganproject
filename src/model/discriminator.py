# -*- coding: utf-8 -*-
import src.utils.config as config
"""
В статье https://arxiv.org/abs/1703.10593 используется PatchGAN в качестве дискриминатора.
Его архитектура имеет виде C64 - C128 - C256 - C512 - Conv(out_channels = 1) - Sigmoid,
где Ck - Conv(in_channels, out_channels = k, kernel_size = 4х4, stride = 2) - InstanceNorm - LeakyRelu(0.2).
Везде, где необходимо используется reflection padding. В первом слое InstanceNorm отсутствует.
В соответсвии со статьей о pix2pix (https://arxiv.org/abs/1611.07004) каждый слой делает downsampling в два раза.
В оригинальной реализации в С512 используется stride = 1, я делаю так-же.
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/162
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.C64 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2,inplace = True),
        )
        self.C128 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 128,kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace = True),
        )
        self.C256 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 256,kernel_size = 4, stride = 2, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace = True),
        )
        self.C512 = nn.Sequential(
            nn.Conv2d(in_channels = 256,out_channels = 512,kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace = True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect'),
            #nn.Sigmoid() - не используется из-за MSELoss - https://arxiv.org/abs/1611.04076
        )
    def forward(self,x):
        x = self.C64(x)
        x = self.C128(x)
        x = self.C256(x)
        x = self.C512(x)
        x = self.out(x)
        return x


def Loss_Disc(D,real_image,fake_image,loss,buffer):
    real_preds = D(real_image)
    real_loss = loss(real_preds, torch.ones_like(real_preds))
    fake_image = buffer.get_image(fake_image.detach())
    fake_preds = D(fake_image)
    fake_loss = loss(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss
