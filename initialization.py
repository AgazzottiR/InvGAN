import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch



# Sets the seed for generating random numbers. Returns a torch.Generator object.


manualSeed = 999
dataroot = r"./data"  # cartella del dataset
workres = 2  # numero dei workers per il dataloader
batch_size = 128  # batch size per l'allenamento è 128 nel paper della DCGAN
image_size = 32  # 32x32 è la dimensione del cifar10
nc = 3  # numero di canali delle immagni in input
nz = 100  # dimesnsione del latent vector
ngf = 64  # size delle feature maps del generatore
ndf = 64  # size delle feature maps del discriminatore
lr = 0.0002  # learining rate
beta1 = 0.5  # iperparametro del Adam optimizer
ngpu = 1  # numero di gpu disponibili
load_chk = False
chk_path = r'./last_checkpoint_cc.pth.tar'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(size=image_size, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
])

'''dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))'''




