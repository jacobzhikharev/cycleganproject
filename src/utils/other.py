# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import src.utils.config as config

#Буфер сделан и работает только для batch_size = 1

class buffer():
    def __init__(self):
        self.size = config.buffer_size
        self.history = []
    def get_image(self,image):
        if len(self.history) < self.size:
            self.history.append(image)
            res = image
        else:
            p = np.random.uniform(0,1)
            if p > 0.5:
                idx = np.random.randint(0,self.size)
                res = self.history[idx]
                self.history[idx] = image
            else:
                res = image
        return res
    def get_buffer(self):
        return self.history

#Ренормализация и решейп изображений
def pr_to_plot(norm_img):
    return np.rollaxis(norm_img.numpy()*0.5+0.5, 0, 3)

#weights initialization like https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# функция для LambdaLR
# lr постоянен первые 100 эпох, затем линейно убывает до нуля
def func2(epoch):
    if epoch<=100:
        return 1.0
    else:
        return np.linspace(config.lr,0,100)[epoch-101]/config.lr
