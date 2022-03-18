from __future__ import print_function

import random
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.nn.functional import one_hot
from initialization import *
from utils import (save_checkpoint, load_checkpoints)
from utils import (weights_init, )
import wandb


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(110, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, y):
        y = y.reshape((y.shape[0], y.shape[1], 1, 1))
        x = torch.cat((x, y), dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main_before_concatenation = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.main_after_concatenation = nn.Sequential(
            nn.Conv2d(74, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):  # x input del discriminatore one hot delle label
        conv_1 = self.main_before_concatenation(x)
        label = torch.tile(torch.reshape(y, [-1, y.shape[-1], 1, 1, ]), [1, 1, conv_1.shape[2], conv_1.shape[3]])
        conv_2_in = torch.cat((conv_1, label), dim=1)
        return self.main_after_concatenation(conv_2_in)

    def forward_feature_extractor(self, x, y):
        fm = [x]
        fm_names = ['input']
        for layer in self.main_before_concatenation.children():
            fm.append(layer(fm[-1]))
            fm_names.append(str(layer))
        fm.append(torch.cat((fm[-1],torch.tile(torch.reshape(y, [-1, y.shape[-1], 1, 1, ]), [1, 1, fm[-1].shape[2], fm[-1].shape[3]])), dim=1))
        fm_names.append("concatenation")
        for layer in self.main_after_concatenation.children():
            fm.append(layer(fm[-1]))
            fm_names.append(str(layer))
        return fm, fm_names



def train_conditional_model_v2():
    c_path = r'last_checkpoint_cc_v2_LC_2.pth.tar'
    ################
    wandb.login(key='bb1abfc16a616453716180cdc3306cf7ce03d891')
    wandb.init(project="my-test-project", entity="riccardoagazzotti")
    wandb.config = {
        "learning_rate": 0.0002,
        "epochs": 50,
        "batch_size": 128
    }
    ##################
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
    y_fixed = one_hot(torch.randint(0,10, (64,), device=device), 10)
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999), eps=10E-8)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), eps=10E-8)

    if True:
        checkpoint = torch.load(c_path)
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
    num_epochs = 30
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            #####################################################
            netD.zero_grad()
            real, y = data
            real, y = real.to(device), y.to(device)
            y = one_hot(y,10)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real,y).view(-1)
            errD_real = criterion(output, label)  # log(D(x))
            errD_real.backward()
            D_x = output.mean().item()  # Errore medio nel classificare immagini reali

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            y = one_hot(torch.randint(0, 10, (b_size,), device=device), 10)
            fake = netG(noise,y)
            label.fill_(fake_label)
            output = netD(fake.detach(),y).view(-1)
            errD_fake = criterion(output, label)  # log(1 - D(G(z)))
            errD_fake.backward()
            D_G_z1 = output.mean().item()  # Errore medio nel calssificare immagini fake
            errD = errD_fake + errD_real
            optimizerD.step()

            ###############################################
            netG.zero_grad()
            label.fill_(real_label)  # fill con label real perchè così usa la fuzione log(x) invece di log(1-x)
            output = netD(fake,y).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()  # Errore medio nel classificare immagini del generatore
            optimizerG.step()
            #############################################

            wandb.log({"errD_real_tot": errD_real,
                       "errD_fake": errD_fake,
                       "errG": errG
                       })
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
                save_checkpoint(checkpoint, filename=r'last_checkpoint_cc_v2_LC_2.pth.tar')
                with torch.no_grad():
                    fake = netG(fixed_noise,y_fixed).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    return G_losses, D_losses, img_list, dataloader


def get_some_outputs(file_pth=r'last_checkpoint_cc_v2_LC_2.pth.tar'):
    netG = Generator()
    netG.to(device)
    netG.load_state_dict(torch.load(file_pth)["state_dict_gen"])
    netG.eval()
    img_list = []

    fixed_noise = torch.randn(100, nz, 1, 1, device=device)
    '''
    y_fixed = one_hot((torch.ones((1,))).to(torch.int64).to(device), 10)
    y_fixed[0,6] = 1
    print(y_fixed)
    with torch.no_grad():
        out = netG(fixed_noise, y_fixed).detach().cpu()
    plt.imshow(np.transpose(out[0], (1, 2, 0)))
    plt.show()'''
    y_fixed = one_hot(torch.from_numpy(np.reshape(np.mgrid[0:10,0:10][0], (-1))).to(torch.int64).to(device), 10)
    a = torch.randint(0, 10, (100, 10))
    y_fixed = torch.where(a >= 5, 1, 0).to(device)
    #fake = torch.zeros((100,3,32,32))
    with torch.no_grad():
        for i in range(10):
            #y_fixed = one_hot((i*torch.ones((10,))).to(torch.int64).to(device),10)
            fake = netG(fixed_noise, y_fixed).detach().cpu()
            #fake[i*10:(i+1)*10] = out
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=10))

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.show()


if __name__ == "__main__":
    get_some_outputs()
    #G_losses, D_losses, img_list, dataloader = train_conditional_model_v2()
    #print_train_results(G_losses,D_losses,img_list,dataloader)