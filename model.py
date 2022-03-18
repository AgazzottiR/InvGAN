from __future__ import print_function
from __future__ import print_function

import random

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from initialization import *
from utils import (save_checkpoint, load_checkpoints)
from utils import (weights_init, )

'''DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal 
distribution with mean=0, stdev=0.02. The weights_init function takes an initialized model as input and reinitializes 
all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria. This function is 
applied to the models immediately after initialization. 
custom weights initialization called on netG and netD'''


# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
# bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # input z e out img
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # vado a (ngf*4) X 8 X 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 5, 2, 2, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # dim qua  (nc) x 32 x 32
        )

    def forward(self, input):
        return F.pad(self.main(input), (0, -1, 0, -1))


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
# padding_mode='zeros', device=None, dtype=None)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 5, 2, 1, bias=True),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def forward_feature_extractor(self, input):
        fm = [input]
        names = ["input"]
        for layer in self.main.children():
            fm.append(layer(fm[-1]))
            names.append(str(layer))
        return fm, names


def train():
    dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workres)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f"Working on device {device}")

    random.seed("Random seed", manualSeed)
    torch.manual_seed(manualSeed)

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.

    # Print the model
    print(netG)
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if load_chk:
        checkpoint = torch.load(chk_path)
        load_checkpoints(checkpoint, netG, netD, optimizerG, optimizerD)
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Training
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Training starting loop...")
    num_epochs = 150
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            #####################################################
            netD.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real).view(-1)
            errD_real = criterion(output, label)  # log(D(x))
            errD_real.backward()
            D_x = output.mean().item()  # Errore medio nel classificare immagini reali

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)  # log(1 - D(G(z)))
            errD_fake.backward()
            D_G_z1 = output.mean().item()  # Errore medio nel calssificare immagini fake
            errD = errD_fake + errD_real
            optimizerD.step()

            ###############################################
            netG.zero_grad()
            label.fill_(real_label)  # fill con label real perchè così usa la fuzione log(x) invece di log(1-x)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()  # Errore medio nel classificare immagini del generatore
            optimizerG.step()
            #############################################

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (i % 300 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                checkpoint = {
                    "state_dict_gen": netG.state_dict(),
                    "optimizer_gen": optimizerG.state_dict(),
                    "state_dict_disc": netD.state_dict(),
                    "optimizer_disc": optimizerD.state_dict()
                }
                # save_checkpoint(checkpoint)
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    return G_losses, D_losses, img_list, dataloader


if __name__ == "__main__":
    train()
