import torch
batch_size = 1#!!! не менять !!!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 256
buffer_size = 50
n_epochs = 10
dir_train_A = './mini/trainA' #Место для хранения изображений одного типа
dir_train_B = './mini/trainB' #Место для хранения изображений другого типа
dir_test_A = './mini/testA' #Место для хранения изображений одного типа(например не учавствовавших в обучении)
dir_test_B = './mini/testB' #Место для хранения изображений другого типа(например не учавствовавших в обучении)
dir_res_A = './res_images/A' #Место для сохранения преобразованых изображений
dir_res_B = './res_images/B' #Место для сохранения преобразованых изображений
dir_saves = './saves' #Место для сохранения промежуточных весов моделей
lam = 10
lr = 0.0002
