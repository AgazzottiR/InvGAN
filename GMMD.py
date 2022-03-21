import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torch.autograd import Variable

root = r'./data'
batch_size = 128
N_INP = 100
N_OUT = 3072
N_GEN_EPOCHS = 20
KERNEL_TYPE = 'rbf'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
workers = 2


class MMDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y, kernel='rbf'):
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


class GMMD(nn.Module):
    def __init__(self, n_start, n_out):
        super(GMMD, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_start, 1000),
            nn.BatchNorm1d(1000),
            nn.Sigmoid(),
            nn.Linear(1000, 600),
            nn.BatchNorm1d(600),
            nn.Sigmoid(),
            nn.Linear(600, 1000),
            nn.BatchNorm1d(1000),
            nn.Sigmoid(),
            nn.Linear(1000, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


def train():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
    ])
    dataset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                             drop_last=True)
    iterations = 0
    Z = torch.randn((5800, batch_size, N_INP))
    gmmd_net = GMMD(N_INP, N_OUT).to(device)
    gmmd_optimizer = optim.RMSprop(gmmd_net.parameters(), lr=0.004)
    lossMMD = MMDLoss()

    for ep in range(N_GEN_EPOCHS):
        avg_loss = 0
        resampling_limit = 300  # From paper
        for idx, (x, _) in enumerate(dataloader):
            iterations += 1
            gmmd_optimizer.zero_grad()
            x = x.view(x.size()[0], -1)
            x = Variable(x).to(device)

            # normal random noise between [0, 1]
            random_noise = Z[idx, :, :]

            samples = Variable(random_noise).to(device)
            gen_samples = gmmd_net(samples)

            loss = lossMMD(x, gen_samples)
            loss.backward()
            gmmd_optimizer.step()
            avg_loss += loss.item()

            if iterations % 300 == 0:
                Z = random_noise = torch.randn((5800, batch_size, N_INP))
        avg_loss /= (idx + 1)
        print(f"GMMD Training: {ep}. epoch completed, average loss: {avg_loss}")
    torch.save(gmmd_net.state_dict(), "params/gmmd.pth")

def get_some_outputs():
    gmmd_net = GMMD(N_INP, N_OUT).to(device)
    gmmd_net.load_state_dict(torch.load(r'params/gmmd.pth'))
    gmmd_net.eval()
    noise = torch.randn((64, N_INP)).to(device)
    with torch.no_grad():
        output_rec = gmmd_net(noise)
        image_noise = vutils.make_grid(output_rec.reshape(64,3,32,32), padding=2, normalize=True)
        plt.imshow(np.transpose(image_noise.cpu(), (1,2,0)))
        plt.show()

if __name__ == "__main__":
    #train()
    get_some_outputs()
