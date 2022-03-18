from __future__ import print_function

import random

import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from PIL import Image
from torch.nn.functional import one_hot
import torchvision.utils as vutils
from model_class_conditional import CDiscriminator, CGenerator
from model import Discriminator, Generator
from utils import (weights_init, )
from utils import (save_checkpoint, load_checkpoints, print_some_train_images)
from measures import (print_train_results, )
from model_class_conditional import (get_noise, )
from initialization import *
from sklearn.cluster import KMeans
from measures import calculate_inception_score
import wandb









if __name__ == "__main__":
    G_losses, D_losses, img_list, trainloader, device = train()
    print_train_results(G_losses, D_losses, img_list, trainloader)
    # G_losses, D_losses, img_list,dataloader = train()
    # print_train_results(G_losses, D_losses, img_list, dataloader)
'''
    transform_inception_v3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    #dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform_inception_v3)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=400, shuffle=True, num_workers=0)
    #img, label = next(iter(dataloader))
    #img = torch.ones_like(img)
    generator = Generator()
    print(generator)
    generator.load_state_dict(torch.load(r'./last_checkpoint.pth.tar')['state_dict_gen'])
    with torch.no_grad():
        generator.eval()
        noise = torch.randn(400, nz, 1, 1)
        images = generator(noise)
        print(images.shape)
        img = torch.zeros((400,3,299,299))
        for i in range(images.shape[0]):
            img[i] = transform_inception_v3(images[i])

    #calculate_inception_score(img,torch.zeros(400))
    Kmeans_DCGAN_feature_extractor()'''
