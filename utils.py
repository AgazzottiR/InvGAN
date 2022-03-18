from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import torch.nn as nn
from initialization import *


def load_checkpoints(checkpoint, netG, netD, optimizerG, optimizerD):
    netG.load_state_dict(checkpoint["state_dict_gen"])
    netD.load_state_dict(checkpoint["state_dict_disc"])
    optimizerG.load_state_dict(checkpoint["optimizer_gen"])
    optimizerD.load_state_dict(checkpoint["optimizer_disc"])


# Stampo alcune immagni di train
def print_some_train_images(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


def save_checkpoint(state, filename="last_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def padding_same(in_height, in_width, kernel_size, stride):
    filter_height, filter_width = kernel_size, kernel_size
    strides = (None, stride, stride)
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width = np.ceil(float(in_width) / float(strides[2]))

    if in_height % strides[1] == 0:
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if in_width % strides[2] == 0:
        pad_along_width = max(filter_width - strides[2], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom
