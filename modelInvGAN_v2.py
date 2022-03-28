import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import optim
from torchvision.transforms import transforms
import torchvision.datasets as dset
import wandb
import matplotlib.pyplot as plt
import numpy as np
from utils import save_checkpoint
from torchmetrics.image.inception import InceptionScore
from measures import calculate_inception_score
class MMDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y, kernel='rbf'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


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
            nn.Conv2d(128, 101, 5, 2, padding=1),
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1)
        )
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = x.reshape((x.shape[0], -1))
        out_z, out_rf = x[:, 0:100], self.sigm(x[:, -1])
        return out_z, out_rf


def train():
    '''wandb.login(key='bb1abfc16a616453716180cdc3306cf7ce03d891')
    wandb.init(project="my-test-project", entity="riccardoagazzotti")
    wandb.config = {
        "learning_rate": 0.0002,
        "epochs": 50,
        "batch_size": 128
    }'''
    c_path = r'params/last_checkpoint_invGAN_6_AIO.pth.tar'
    batch_size = 128
    image_size = 32
    workers = 2
    p_load = True
    ngpu = 1
    num_epochs = 50
    nz = 512
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

    netM = MappingNetwork()
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
    optimizerM = optim.Adam(netM.parameters(), lr=lr, betas=(beta1, beta2), eps=10E-8)

    if (device == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    try:
        if p_load:
            checkpoint = torch.load(c_path)
            netM.load_state_dict(checkpoint['state_dict_m'])
            netG.load_state_dict(checkpoint['state_dict_gen'])
            netD.load_state_dict(checkpoint['state_dict_disc'])
            optimizerD.load_state_dict(checkpoint['optimizer_disc'])
            optimizerG.load_state_dict(checkpoint['optimizer_gen'])
            optimizerM.load_state_dict(checkpoint['optimizer_m'])
        else:
            netG.apply(weights_init)  # Init with the initialization proposed by DCGAN
            netD.apply(weights_init)
            netM.apply(weights_init)
    except:
        pass

    lossL2 = nn.MSELoss()
    lossMMD = MMDLoss()
    lossBCE = nn.BCELoss()

    l = 0.

    previous_batch = torch.zeros((0, 100)).to(device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real = data[0].to(device)
            label = torch.zeros(real.size(0)).to(device)

            # Discriminatore
            netM.zero_grad()
            netD.zero_grad()
            label.fill_(real_label)
            output_z_real, output_real = netD(real)
            noise = torch.randn((real.size(0), nz), device=device)
            errD_real = lossBCE(output_real, label)
            # errD_real.backward()
            (errD_real).backward()

            m_noise = netM(noise)
            if previous_batch.shape[0] > 0:
                index = torch.randperm(previous_batch.shape[0])
                previous_batch = previous_batch[index]
                m_noise = torch.cat((m_noise[real.size(0) // 2:], previous_batch[:real.size(0) // 2]))

            if previous_batch.shape[0] >= real.size(0)*5:
                previous_batch = output_z_real.detach()
            else:
                previous_batch = torch.cat((previous_batch, output_z_real.detach()))




            label.fill_(fake_label)
            fake = netG(m_noise[:, :, None, None].detach())
            output_z_fake, output_fake = netD(fake.detach())
            output_fake = output_fake.reshape(output_fake.shape[0])

            errD_MMD = lossMMD(m_noise.reshape(m_noise.shape[0], m_noise.shape[1]),output_z_real)  # Loss MMD out_real Noise (in fake)
            errD_reconstruction = lossL2(m_noise.reshape(m_noise.shape[0:2]),output_z_fake)  # Loss L2 first block fake data flow
            errD_fake = lossBCE(output_fake, label)  # Loss of gan

            (errD_fake + errD_reconstruction * l).backward()

            errD = errD_fake + errD_real
            optimizerM.step()
            optimizerD.step()

            # Generatore
            netG.zero_grad()
            netM.zero_grad()
            label.fill_(real_label)
            output_z, output_gen_clf = netD(fake)
            output_gen_clf = output_gen_clf.reshape(output_gen_clf.shape[0])
            errG_real = lossBCE(output_gen_clf, label)  # Loss GAN for generator

            errG_L2 = lossL2((m_noise[real.size(0) // 2:].reshape((m_noise.shape[0] - real.size(0) // 2), m_noise.shape[1])).detach(),output_z[real.size(0) // 2:])
            errG = errG_real + errG_L2
            # Feature loss here
            errG.backward()
            optimizerG.step()

            '''wandb.log({
                "Err_Disc_Gan_real": errD_real,
                "Err_Disc_Gan_fake": errD_fake,
                "Err_D": errD,
                "Err_Gen": errG,
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
                    "optimizer_m": optimizerM.state_dict(),
                    "state_dict_m": netM.state_dict()
                }
                save_checkpoint(checkpoint, filename=r'params/last_checkpoint_invGAN_6_AIO_2.pth.tar')
            l = l + 0.05 if l < 1 else 1


def get_some_output():
    paths = [
             r'params/last_checkpoint_invGAN_5_BCE_MN.pth.tar',
             r'params/last_checkpoint_invGAN_6_AIO.pth.tar',
             r'params/last_checkpoint_invGAN_6_AIO_2.pth.tar',
             ]
    netG = Generator()
    netD = Discriminator()
    netM = MappingNetwork()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    netG.to(device)
    netD.to(device)
    netM.to(device)

    for path in paths:
        noise = torch.randn((64, 512)).to(device)
        netG.load_state_dict(torch.load(path)['state_dict_gen'])
        netD.load_state_dict(torch.load(path)['state_dict_disc'])
        netM.load_state_dict(torch.load(path)['state_dict_m'])
        netM.eval()
        netG.eval()
        netD.eval()
        with torch.no_grad():
            noise = netM(noise)
            noise = noise[:, :, None, None]
            output_n = netG(noise)
            out_rec, _ = netD(output_n)
            output_rec = netG(out_rec[:,:,None,None])
            loss = nn.MSELoss()
        print(f"Loss for noise {loss(noise,out_rec)}")
        images_rec = vutils.make_grid(output_rec, padding=2, normalize=True)
        image_noise = vutils.make_grid(output_n, padding=2, normalize=True)
        #calculate_inception_score(output_n.cpu(),torch.zeros(output_n.shape[0]))

        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Image noise")
        plt.imshow(
            np.transpose(vutils.make_grid(image_noise.cpu(), padding=2, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Images reconstructed")
        plt.imshow(np.transpose(images_rec.cpu(), (1, 2, 0)))
        plt.show()


if __name__ == "__main__":
    get_some_output()
    '''
    with torch.autograd.set_detect_anomaly(True):
        train()'''
