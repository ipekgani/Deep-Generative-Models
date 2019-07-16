import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torch.autograd import Variable
import pickle
from scipy.interpolate import griddata
plt.style.use('seaborn-darkgrid')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.seq_forward = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        return self.seq_forward(z)

    def loss(self, fake):
        return - torch.mean(torch.log(fake))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq_forward = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        return self.seq_forward(img)

    def loss(self, fake, real):
        return - torch.mean(torch.log(real) + torch.log(1.-fake))


def show(img, epoch_num):
    npimg = img.detach().cpu().numpy()
    npimg = (npimg * 255).astype(np.uint8)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.title(epoch_num)
    plt.axis('off')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    # fig1.savefig('0_gan_' + str(epoch_num) +'.png')

def loss(prediction, target, criterion):
    return criterion(prediction, target).sum(dim=-1).mean()

def plot_loss(loss_curves, write=False):
    plt.plot(loss_curves['discri'], label='discriminator', color='b')
    plt.plot(loss_curves['gen'], label='generator', color='r')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend()

    if write: fig1 = plt.gcf()
    plt.show()
    if write:
        plt.draw()
        fig1.savefig('0_GAN_losses.png')

def normalize_imgs(x):
    (mins,_) = x.min(dim=-1)
    (maxs,_) = x.max(dim=-1)
    mins, maxs = mins.unsqueeze(dim=-1), maxs.unsqueeze(dim=-1)
    return 2*(x-mins)/(maxs-mins)-1

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    device = torch.device(torch.cuda.current_device())
    print('Starting training')
    z_for_show = torch.randn((100, 100)).to(device).detach()
    loss_curves = {'discri': [], 'gen': []}
    for epoch in range(args.n_epochs):
        epoch_gen_discri = {'discri':[], 'gen':[]}
        for i, (imgs, _) in enumerate(dataloader):
            current_batch_size = imgs.shape[0]
            if torch.cuda.is_available():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            imgs = imgs.to(device).squeeze().reshape([current_batch_size, 28*28])
            imgs = Variable(normalize_imgs(imgs))

            # Train Generator
            # ---------------
            noise = Variable(torch.randn((current_batch_size, 100)))
            gen_imgs = generator(z=noise)
            loss_G = generator.loss(discriminator(gen_imgs))
            loss_G.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            noise = Variable(torch.randn((current_batch_size, 100)))
            gen_imgs = generator(z=noise)

            loss_D = discriminator.loss(fake=discriminator(gen_imgs.detach()),real=discriminator(imgs))
            loss_D.backward()
            optimizer_D.step()

            epoch_gen_discri['discri'].append(loss_D.detach().item())
            epoch_gen_discri['gen'].append(loss_G.detach().item())

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                gen_imgs = generator(z=z_for_show).reshape(100, 1, 28, 28).detach()

                save_image(gen_imgs, 'images/{}.png'.format(batches_done), nrow=10, normalize=True)
                show(make_grid(gen_imgs, nrow=10), str(batches_done) + ' '+ str(epoch))

            torch.set_default_tensor_type('torch.FloatTensor')

        loss_curves['gen'].append(np.mean(epoch_gen_discri['gen']))
        loss_curves['discri'].append(np.mean(epoch_gen_discri['discri']))

        print(f"[Epoch {epoch}] losses: G: {loss_curves['gen'][-1]} "
              f"D: {loss_curves['discri'][-1]}")
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), 'saved_gans/gan_' + str(epoch)+ '.model')
            with open('GAN.pickle', 'wb') as handle:
                pickle.dump(loss_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plot_loss(loss_curves)

    plot_loss(loss_curves, write=True)

def load_and_plot():
    if os.path.isfile('results/saved_gans/GAN.pickle'):
        loss_curves = pickle.load(open('results/saved_gans/GAN.pickle', 'rb'))
        plot_loss(loss_curves, write=True)


def load_and_interpolate(n=8):
    torch.manual_seed(3)
    device = torch.cuda.current_device()
    generator = Generator().cuda()
    generator.load_state_dict(torch.load('results/saved_gans/gan_145.model'))
    generator.eval()

    z1 = torch.randn((1, 100)).detach() #+ 0.7
    z2 = torch.randn((1, 100)).detach() #- 0.7

    z_between = np.zeros((n, 100))
    for d in range(100):
        dummy = torch.Tensor(np.linspace(z1[:, d], z2[:, d], num=n))
        z_between[:, d] = torch.Tensor(np.linspace(z1[:, d], z2[:, d], num=n)).squeeze()
    Z = torch.cat((z1, torch.Tensor(z_between), z2), dim=0).detach().to(device)

    imgs = generator(Z)
    show(make_grid(imgs.reshape(n+2, 1, 28, 28), nrow=n+2), ' ')
    save_image(imgs.reshape(n+2, 1, 28, 28), 'interpolated.png', nrow=n+2, normalize=True)

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    os.makedirs('saved_gans', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, ), std=(0.5,))])),
        batch_size=args.batch_size, shuffle=True)
    # Initialize models and optimizers
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=150,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, ), std=(0.5,))])),
        batch_size=args.batch_size, shuffle=True)
    main()
    # load_and_plot()
    # load_and_interpolate()