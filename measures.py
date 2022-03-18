from math import floor

import matplotlib as plt
import sklearn.metrics
import torch.nn.parallel
from torch.nn.functional import one_hot
import torch.utils.data
import torchvision.utils as vutils
from numpy import exp
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from sklearn.cluster import KMeans

from initialization import *
from model_class_conditional_v2 import Discriminator, Generator


# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, label, n_split=10, eps=1E-16):
    # load cifar10 images
    print('loaded', images.shape)
    # load inception v3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.eval()
    # enumerate splits of images/predictions

    scores = list()
    n_part = floor(images.shape[0] / n_split)
    with torch.no_grad():
        for i in range(n_split):
            # retrieve images
            ix_start, ix_end = i * n_part, (i + 1) * n_part
            subset = images[ix_start:ix_end]
            label_ss = label[ix_start:ix_end]
            for l in label_ss.unique():
                print(f"label {l} is present {(label_ss == l).sum()}")
            # convert from uint8 to float32
            subset = subset.to(torch.float32)
            # scale images to the required size
            # pre-process images, scale to [-1,1]
            # predict p(y|x)
            p_yx = model(subset)
            activation = torch.nn.Softmax(dim=1)
            p_yx = activation(p_yx)
            # calculate p(y)
            p_yx = p_yx.numpy()
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = mean(sum_kl_d)
            # undo the log
            is_score = exp(avg_kl_d)
            # store
            scores.append(is_score)
            print(f"Iteration score {i}, {is_score}")
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    print('score', is_avg, is_std)
    return is_avg, is_std


def print_train_results(G_losses, D_losses, img_list, dataloader):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    # ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


def inference_DCGAN_discriminator(x, label=None, parampath=r'./last_checkpoint_cc_v2_LC.pth.tar', idx=-3):
    model = Discriminator().to(device)
    checkpoint = torch.load(parampath)
    model.load_state_dict(checkpoint['state_dict_disc'])
    model.eval()
    x = x.to(device)
    y = None
    if label is not None:
        y = one_hot(label, 10).to(device)
    with torch.no_grad():
        fm, fm_names = model.forward_feature_extractor(x, y)
    return fm[idx], fm_names[idx]


def Kmeans_DCGAN_feature_extractor():
    ############Parameters for Kmeans######
    batch_size_kmeans = 4000
    transform_km = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
    ])

    trainset = dset.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_kmeans, shuffle=True, num_workers=0)

    testset = dset.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_kmeans, shuffle=False, num_workers=0)

    classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # train
    netG = Generator()
    netG.to(device)
    netG.load_state_dict(torch.load(r'last_checkpoint_cc_v2_LC.pth.tar')["state_dict_gen"])
    netG.eval()

    fixed_noise = torch.randn(4000, nz, 1, 1, device=device)
    label = torch.randint(0, 10, (4000,), device=device).to(torch.int64)
    y_fixed = one_hot(label, 10)

    with torch.no_grad():
        images = netG(fixed_noise, y_fixed).detach().cpu()
        img = torch.zeros((4000, 3, 32, 32))
        for i in range(images.shape[0]):
            img[i] = transform_km(images[i])

    # img, label = next(iter(trainloader))
    print(label.shape)
    result, layer = inference_DCGAN_discriminator(img, label)
    result = result.reshape((batch_size_kmeans, -1))
    print(f"Computing on layer {layer}")
    idxs = []
    for i in idxs:
        fm, layer = inference_DCGAN_discriminator(img, idx=i)
        fm = fm.reshape((batch_size_kmeans, -1))
        result = torch.cat((result, fm), dim=1)
        print(f"Computing on layer {layer}")
    print(f"Kmeans on {result.shape}")
    kmeans = KMeans(n_clusters=10, random_state=0).fit(result.cpu().numpy())
    # test
    netG = Generator()
    netG.to(device)
    netG.load_state_dict(torch.load(r'last_checkpoint_cc_v2_LC.pth.tar')["state_dict_gen"])
    netG.eval()

    fixed_noise = torch.randn(4000, nz, 1, 1, device=device)
    label = torch.randint(0, 10, (4000,), device=device).to(torch.int64)
    y_fixed = one_hot(label, 10)

    with torch.no_grad():
        images = netG(fixed_noise, y_fixed).detach().cpu()
        img = torch.zeros((4000, 3, 32, 32))
        for i in range(images.shape[0]):
            img[i] = transform_km(images[i])

    #img, label = next(iter(testloader))
    result, layer = inference_DCGAN_discriminator(img, label)
    result = result.reshape((batch_size_kmeans, -1))
    for i in idxs:
        fm, layer = inference_DCGAN_discriminator(img, idx=i)
        fm = fm.reshape((batch_size_kmeans, -1))
        result = torch.cat((result, fm), dim=1)

    prediction = kmeans.predict(result.cpu().numpy())
    accuracy = sklearn.metrics.accuracy_score(label.cpu(), prediction)
    print(f"Accuracy {accuracy}\n")


if __name__ == "__main__":
    Kmeans_DCGAN_feature_extractor()

    '''
    transform_inception_v3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2467, 0.2431, 0.2611)),
    ])

    netG = Generator()
    netG.to(device)
    netG.load_state_dict(torch.load(r'last_checkpoint_cc_v2_LC.pth.tar')["state_dict_gen"])
    netG.eval()
    img_list = []

    fixed_noise = torch.randn(400, nz, 1, 1, device=device)
    label = torch.randint(0, 10, (400,), device=device).to(torch.int64)
    y_fixed = one_hot(label, 10)

    with torch.no_grad():
        images = netG(fixed_noise, y_fixed).detach().cpu()
        img = torch.zeros((400, 3, 299, 299))
        for i in range(images.shape[0]):
            img[i] = transform_inception_v3(images[i])
    calculate_inception_score(img, label)
'''