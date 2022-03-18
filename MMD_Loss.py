import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2Distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2Distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2Distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[:batch_size, :batch_size]
        XY = kernels[:batch_size, :batch_size]
        YX = kernels[:batch_size, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class MMDLoss(nn.Module):
    def __init__(self, alpha=1):
        super(MMDLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        B = x.shape[0]

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(- self.alpha * (rx.t() + rx - 2 * xx))
        L = torch.exp(- self.alpha * (ry.t() + ry - 2 * yy))
        P = torch.exp(- self.alpha * (rx.t() + ry - 2 * zz))

        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))

        return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)


def debug():
    source = torch.randn((500, 100))
    target = torch.randn((500, 100))
    loss = MMDLoss()
    print(loss(source, target))


if __name__ == "__main__":
    debug()
