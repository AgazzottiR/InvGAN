import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from utils import load_checkpoints, weights_init
from MMD_Loss import MMDLoss
import torchvision.utils as vutils
from utils import save_checkpoint
import wandb


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 2, padding=1),
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

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 101)
        )
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = x.reshape((x.shape[0], -1))
        out = self.linear(x)
        out_z, out_rf = out[:, 0:100], self.sigm(out[:, -1])
        return out_z, out_rf

    def forward_feature_extractor(self, x):
        fm = [x]
        fm_names = ['input']
        for layer in self.main.children():
            fm.append(layer(fm[-1]))
            fm_names.append(str(layer))
        fm[-1] = fm[-1][:, :, 0, 0]
        fm_names.append('reshape')
        for layer in self.linear.children():
            fm.append(layer(fm[-1]))
            fm_names.append(str(layer))
        fm.append([fm[-1][:, 0:100], self.sigm(fm[-1][:, -1])])
        fm_names.append('out as list')
        return fm, fm_names


class MappingNet(nn.Module):
    def __init__(self):
        super(MappingNet, self).__init__()
        self.main = nn.Sequential(
            nn.Identity()
        )

    def forward(self, x):
        return self.main(x)


def train():
    '''
    wandb.login(key='bb1abfc16a616453716180cdc3306cf7ce03d891')
    wandb.init(project="my-test-project", entity="riccardoagazzotti")
    wandb.config = {
        "learning_rate": 0.0002,
        "epochs": 200,
        "batch_size": 128
    }'''
    c_path = r'last_checkpoint_invGAN.pth.tar'
    batch_size = 128
    image_size = 32
    workers = 2
    p_load = False
    ngpu = 1
    num_epochs = 100
    nz = 100
    real_label = 1.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fake_label = 0.
    img_list = []
    fixed_noise = torch.randn((64, nz, 1, 1), device=device)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
    ])

    dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)

    netM = MappingNet()
    netD = Discriminator()
    netG = Generator()

    netM.to(device)
    netD.to(device)
    netG.to(device)

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2), eps=10E-8)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2), eps=10E-8)
    # optimizerM = optim.Adam(netM.parameters(), lr=lr, betas=(beta1, beta2), eps=10E-8)

    if (device == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    if p_load:
        checkpoint = torch.load(c_path)
        netM.load_state_dict(checkpoint['state_dict_m'])
        netG.load_state_dict(checkpoint['state_dict_gen'])
        netD.load_state_dict(checkpoint['state_dict_disc'])
        optimizerD.load_state_dict(checkpoint['optimizer_disc'])
        optimizerG.load_state_dict(checkpoint['optimizer_gen'])
        # optimizerM.load_state_dict(checkpoint['optimizer_m']) --> mapping still no parameters
    else:
        netG.apply(weights_init)  # Init with the initialization proposed by DCGAN
        netD.apply(weights_init)
        netM.apply(weights_init)

    lossL2 = nn.MSELoss()
    lossMMD = MMDLoss()
    lossBCE = nn.BCELoss()

    previous_batch = torch.zeros((0, 100)).to(device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real = data[0].to(device)
            label = torch.zeros(real.size(0)).to(device)
            if real.size(0) != batch_size:
                continue

            if previous_batch.shape[0] > 0:
                noise = torch.cat((torch.randn((real.size(0) // 2, nz, 1, 1), device=device),
                                   previous_batch[(real.size(0) // 2):real.size(0), :, None, None]), dim=0)
            else:  # Genero minibatch misto
                noise = torch.randn((real.size(0), nz, 1, 1), device=device)

            # Now noise is splitted in noise_fake e noise real
            assert noise.shape[0] == label.shape[0] == real.shape[0]

            # Discriminatore
            netD.zero_grad()
            label.fill_(real_label)
            output_z_real, output_real = netD(real)

            # Storing minibatch
            if previous_batch.shape[0] >= batch_size * 5:
                previous_batch = torch.zeros((0, 100)).to(device)
            previous_batch = torch.cat((previous_batch, output_z_real.detach()), dim=0)
            previous_batch = previous_batch[torch.randperm(previous_batch.size(0))]  # shuffle

            errD_MMD = lossMMD(noise.reshape(noise.shape[0], noise.shape[1]),
                               output_z_real) * 0.2  # Loss MMD out_real Noise (in fake)
            errD_reconstruction = lossL2(noise[:real.size(0)].reshape(noise.shape[0:2]).detach(),
                                         output_z_real[:real.size(0)])  # Loss L2 first block fake data flow
            errD_real = lossBCE(output_real, label)

            (errD_MMD + errD_reconstruction + errD_real).backward()

            label.fill_(fake_label)
            fake = netG(noise)
            output_z_fake, output_fake = netD(fake.detach())
            output_fake = output_fake.reshape(output_fake.shape[0])

            errD_fake = lossBCE(output_fake, label)  # Loss of gan
            errD_fake.backward()
            errD = errD_MMD + errD_reconstruction + errD_real + errD_fake
            optimizerD.step()

            # Generatore
            netG.zero_grad()
            label.fill_(real_label)
            output_z, output_gen_clf = netD(fake)
            output_gen_clf = output_gen_clf.reshape(output_gen_clf.shape[0])

            errG_real = lossBCE(output_gen_clf, label)  # Loss GAN for generator

            errG_L2 = lossL2(noise[real.size(0) // 2:].reshape((noise.shape[0] - real.size(0) // 2), noise.shape[1]),
                             output_z[real.size(0) // 2:])

            errG = errG_L2 + errG_real
            # Feature loss here
            errG.backward()
            optimizerG.step()
            '''wandb.log({
                "Err_Disc_Gan_real" : errD_real,
                "Err_Disc_Gan_fake": errD_fake,
                "Err_Disc_MMD": errD_MMD,
                "Err_Disc": errD,
                "Err_Gen": errG,
                "Err_Gen_GAN": errG_real,
                "Err_Gen_L2": errG_L2
            })'''
            # Stats from here
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t' % (
                    epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(),))

            if (i % 300 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                checkpoint = {
                    "state_dict_gen": netG.state_dict(),
                    "optimizer_gen": optimizerG.state_dict(),
                    "state_dict_disc": netD.state_dict(),
                    "optimizer_disc": optimizerD.state_dict(),
                    # "optimizer_m": optimizerM.state_dict(),
                    "state_dict_m": netM.state_dict()
                }
                save_checkpoint(checkpoint, filename=r'params/last_checkpoint_invGAN_2.pth.tar')
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))


def get_some_output():
    netG = Generator()
    netD = Discriminator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netG.to(device)
    netD.to(device)
    noise = torch.randn((64, 100, 1, 1)).to(device)
    netG.load_state_dict(torch.load(r'params/last_checkpoint_invGAN.pth.tar')['state_dict_gen'])
    netD.load_state_dict(torch.load(r'params/last_checkpoint_invGAN.pth.tar')['state_dict_disc'])
    netG.eval()
    netD.eval()

    with torch.no_grad():
        output_n = netG(noise)
        out_rec, _ = netD(output_n)
        loss = nn.MSELoss()
        val_loss = loss(out_rec, noise.reshape(noise.shape[0], noise.shape[1]))
        output_rec = netG(out_rec[:, :, None, None])
    images_rec = vutils.make_grid(output_rec, padding=2, normalize=True)
    image_noise = vutils.make_grid(output_rec, padding=2, normalize=True)

    print(f"MSE is {val_loss}")

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Image noise")
    plt.imshow(
        np.transpose(vutils.make_grid(image_noise.cpu(), padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Images reconstructed")
    plt.imshow(np.transpose(images_rec.cpu(), (1, 2, 0)))
    plt.show()


def debug():
    x = torch.randn((100, 3, 32, 32))
    label = torch.randn((100, 100))
    rf = torch.randn(100)

    disc = Discriminator()
    opt = optim.Adam(disc.parameters())
    fm, fm_names = disc.forward_feature_extractor(x)

    for a, b in zip(fm, fm_names):
        print(f"Risultato di shape {a.shape} per il layer {b} del discriminatore")


if __name__ == "__main__":
    # get_some_output()
    train()
