from __future__ import print_function
from __future__ import print_function

import random

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.nn.functional import one_hot

import wandb
from initialization import *
from utils import (save_checkpoint, load_checkpoints)
from utils import (weights_init, )

c = 10
nz = 100


def get_noise(batch_size_cc, device='cuda:0'):
    noise = torch.zeros((batch_size_cc, 100 * 10), device=device)
    f_l = []
    for v in range(batch_size_cc):
        cl = torch.randint(0, 9, (1,))
        f_l.append(cl.item())
        noise[v, (cl.item() * 100):(cl.item() + 1) * 100] = torch.randn((100,))
    return noise, f_l


class CGenerator(nn.Module):
    def __init__(self):
        super(CGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, 2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 5, 1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.reshape((-1, 256, 8, 8))
        x = self.conv(x)
        x = F.pad(x, (-1, 0, -1, 0))
        return x


# Padding 4
class CDiscriminator(nn.Module):
    def __init__(self):
        super(CDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 5, 2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256 * 4 * 4, 12),
        )
        self.sm = nn.Softmax(dim=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.shape[0], -1))
        x = self.mlp(x)
        x_cl = self.sm(x[:, 0:10])
        x_rf = self.sm(x[:, -2:])
        return torch.cat((x_cl, x_rf), dim=1)


def train_classCond():
    # initialization
    wandb.login(key='bb1abfc16a616453716180cdc3306cf7ce03d891')
    wandb.init(project="my-test-project", entity="riccardoagazzotti")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 128
    }

    batch_size_cc = 32
    num_epochs = 96
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(f"Working on device {device}")
    lr_cc = 0.00003

    random.seed("Random seed", manualSeed)
    torch.manual_seed(manualSeed)

    trainset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_cc, shuffle=True, num_workers=0)

    testset = dset.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_cc, shuffle=False, num_workers=0)

    netG = CGenerator().to(device)
    netD = CDiscriminator().to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    print(netG)
    print(netD)

    criterion = nn.CrossEntropyLoss()
    fixed_noise, fn_l = get_noise(batch_size_cc, device)

    optimizerD = optim.Adam(netD.parameters(), lr=lr_cc, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_cc, betas=(beta1, 0.999))

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

    for epoch in range(num_epochs):
        # Discriminator
        for i, data in enumerate(trainloader, 0):
            netD.zero_grad()
            real, label = data
            real = real.to(device)
            label = label.to(device)
            label = one_hot(label, 12)
            label[:, 10] = 0.
            label[:, 11] = 1.  # set real label
            label, real = label.type(torch.float32), real.type(torch.float32)
            b_size = real.size(0)
            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise, f_l = get_noise(batch_size_cc, device)

            fake = netG(noise)
            label = one_hot(torch.Tensor(f_l).to(torch.int64), 12).to(torch.float32).to(device)  # set to false
            label[:, 10] = 1.
            label[:, 11] = 0.
            output = netD(fake.detach())
            errD_fake = criterion(output.type(torch.float32).to(device), label)  # log(1 - D(G(z)))
            errD_fake.backward()

            errD = errD_fake + errD_real
            optimizerD.step()

            netG.zero_grad()
            label[:, 10] = 1.
            label[:, 11] = 0.
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            wandb.log({
                       "errD_real_tot": errD_real,
                       "errD_fake": errD_fake,
                       "errG": errG
                       })
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, num_epochs, i, len(trainloader),
                         errD.item(), errG.item(),))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 1500 == 0) or ((epoch == num_epochs - 1) and (i == len(trainloader) - 1)):
                checkpoint = {
                    "state_dict_gen": netG.state_dict(),
                    "optimizer_gen": optimizerG.state_dict(),
                    "state_dict_disc": netD.state_dict(),
                    "optimizer_disc": optimizerD.state_dict()
                }
                save_checkpoint(checkpoint, filename=r'./last_checkpoint_cc_VERSIONE_CROSS_ENTROPY.pth.tar')
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    return G_losses, D_losses, img_list, trainloader, device

if __name__ == "__main__":
    train_classCond()