# -*- coding: utf-8 -*-
import src.utils.config as config

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt
import os
from PIL import Image
'''
Класс для пар картинок
Если изображений одного типа больше, то в пару к ним ставятся случайные
'''
class paired_dataset(Dataset):
    def __init__(self,dir_A,dir_B,transforms = None):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transforms = transforms
        self.data_A = []
        self.data_B = []
        for _, _, images_list in os.walk(self.dir_A):
            for image_name in images_list:
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    self.data_A.append(image_name)
        for _, _, images_list in os.walk(self.dir_B):
            for image_name in images_list:
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    self.data_B.append(image_name)
        self.data_A = sorted(self.data_A)#Для проверки модели я использую датасет facades.
        self.data_B = sorted(self.data_B)#Возможно качество на нем немного улучшается если подавать правильные пары(фотографию и разметку)
    def __getitem__(self,index):
        idxA = index
        idxB = index
        if index >= len(self.data_A):
            idxA = np.random.randint(len(self.data_A))
        elif index >= len(self.data_B):
            idxB = np.random.randint(len(self.data_B))
        path_A = os.path.join(self.dir_A,self.data_A[idxA])
        path_B = os.path.join(self.dir_B,self.data_B[idxB])
        image_A = Image.open(path_A).convert('RGB')#there are some black and white images
        image_B = Image.open(path_B).convert('RGB')
        if self.transforms:
            image_A = self.transforms(image_A)
            image_B = self.transforms(image_B)
        return image_A,image_B
    def __len__(self):
        return max(len(self.data_A),len(self.data_B))

'''
Класс для пар картинок
Если изображений одного типа больше, то в пару к ним ставятся случайные
Эта версия загружает данные сразу на gpu
'''
class paired_dataset_updated(Dataset):
    def __init__(self,dir_A,dir_B,transforms = None,requires_flip = True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transforms = transforms
        self.data_A = []
        self.data_B = []
        self.listed_A = []
        self.listed_B = []
        self.requires_flip = requires_flip
        for _, _, images_list in os.walk(self.dir_A):
            for image_name in images_list:
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    self.data_A.append(image_name)
        for _, _, images_list in os.walk(self.dir_B):
            for image_name in images_list:
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    self.data_B.append(image_name)
        self.data_A = sorted(self.data_A)#Для проверки модели я использую датасет facades.
        self.data_B = sorted(self.data_B)#Возможно качество на нем немного улучшается если подавать правильные пары(фотографию и разметку)
        for path_img in self.data_A:
            path_A = os.path.join(self.dir_A,path_img)
            image = Image.open(path_A).convert('RGB')
            image = self.transforms(image)
            image = image.to(config.device)
            self.listed_A.append(image)
        for path_img in self.data_B:
            path_B = os.path.join(self.dir_B,path_img)
            image = Image.open(path_B).convert('RGB')
            image = self.transforms(image)
            image = image.to(config.device)
            self.listed_B.append(image)
    def __getitem__(self,index):
        idxA = index
        idxB = index
        if index >= len(self.data_A):
            idxA = np.random.randint(len(self.data_A))
        elif index >= len(self.data_B):
            idxB = np.random.randint(len(self.data_B))
        image_A = self.listed_A[idxA]
        image_B = self.listed_B[idxB]
        if self.requires_flip:
            image_A = tt.RandomHorizontalFlip(p = 0.5)(image_A)
            image_B = tt.RandomHorizontalFlip(p = 0.5)(image_B)
        return image_A,image_B
    def __len__(self):
        return max(len(self.data_A),len(self.data_B))

'''
Класс для пар картинок
Если изображений одного типа больше, то в пару к ним ставятся случайные
Эта версия загружает данные сразу на gpu
Используется только для генерации и возвращает вместе с изображениями их имена
'''

class paired_dataset_updated_str(Dataset):#This version load all images directly to device(gpu)
    def __init__(self,dir_A,dir_B,transforms = None,requires_flip = True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transforms = transforms
        self.data_A = []
        self.data_B = []
        self.listed_A = []
        self.listed_B = []
        self.requires_flip = requires_flip
        for _, _, images_list in os.walk(self.dir_A):
            for image_name in images_list:
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    self.data_A.append(image_name)
        for _, _, images_list in os.walk(self.dir_B) or image_name.endswith(".png"):
            for image_name in images_list:
                if image_name.endswith(".jpg"):
                    self.data_B.append(image_name)
        self.data_A = sorted(self.data_A)#Для проверки модели я использую датасет facades.
        self.data_B = sorted(self.data_B)#Возможно качество на нем немного улучшается если подавать правильные пары(фотографию и прямоугольники)
        for path_img in self.data_A:
            path_A = os.path.join(self.dir_A,path_img)
            image = Image.open(path_A).convert('RGB')
            image = self.transforms(image)
            image = image.to(config.device)
            self.listed_A.append(image)
        for path_img in self.data_B:
            path_B = os.path.join(self.dir_B,path_img)
            image = Image.open(path_B).convert('RGB')
            image = self.transforms(image)
            image = image.to(config.device)
            self.listed_B.append(image)
    def __getitem__(self,index):
        idxA = index
        idxB = index
        if index >= len(self.data_A):
            idxA = np.random.randint(len(self.data_A))
        elif index >= len(self.data_B):
            idxB = np.random.randint(len(self.data_B))
        image_A = self.listed_A[idxA]
        image_B = self.listed_B[idxB]
        if self.requires_flip:
            image_A = tt.RandomHorizontalFlip(p = 0.5)(image_A)
            image_B = tt.RandomHorizontalFlip(p = 0.5)(image_B)
        return self.data_A[idxA],self.data_B[idxB],image_A,image_B
    def __len__(self):
        return max(len(self.data_A),len(self.data_B))
