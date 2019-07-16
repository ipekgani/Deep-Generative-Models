import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.autograd import Variable
from datasets.bmnist import bmnist
import numpy as np
from torchvision.utils import save_image
import pickle
from scipy.stats import norm

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        self.seq_forward = nn.Sequential(
            nn.Linear(28*28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_dim, z_dim)
        self.z_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        hidden_out = self.seq_forward(input)
        mean = self.z_mean(hidden_out)
        logvar = self.z_logvar(hidden_out)
        std = torch.sqrt(torch.exp(logvar))
        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        self.seq_forward = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28*28),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.
        Returns mean with shape [batch_size, 784].
        """
        mean = self.seq_forward(input)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.recon_loss = nn.BCELoss(reduction='none')

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder.forward(input)
        z = self.sample_z(mean, std)
        mean_out = self.decoder.forward(z)
        # out = torch.bernoulli(mean_out)
        average_negative_elbo, elbo_recon, elbo_kl = self.elbo(mean_out, input, mean, std)
        return average_negative_elbo, elbo_recon, elbo_kl

    def sample_z(self, mean, std):
        noise = Variable(torch.randn(mean.size()))
        return (noise * std) + mean

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn((n_samples, self.z_dim))
        im_means = self.decoder.forward(z.detach())
        sampled_ims = torch.bernoulli(im_means).detach()

        return sampled_ims, im_means

    def elbo(self, output, input, mean, std):
        rec_loss = (torch.sum(self.recon_loss(output, input),dim=-1)).mean()
        kl_loss = (- 0.5* torch.sum(1 + torch.log(std**2)-mean**2-std**2, dim=-1)).mean()
        return rec_loss + kl_loss, rec_loss, kl_loss


def epoch_iter(model, data, optimizer, device):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    losses = {'recon': [], 'kl':[], 'elbo':[]}
    if model.training:
        for i_batch, sample_batched in enumerate(data):
            input_flat = sample_batched.reshape(sample_batched.shape[0], 28*28).to(device);

            avg_elbo, elbo_recon, elbo_kl = model.forward(input_flat)
            optimizer.zero_grad()
            avg_elbo.backward()
            optimizer.step()
            losses['kl'].append(elbo_kl.item())
            losses['elbo'].append(avg_elbo.item())
            losses['recon'].append(elbo_recon.item())

    else: # validation
         for i_batch, sample_batched in enumerate(data):
            input_flat = sample_batched.reshape(sample_batched.shape[0], 28*28).to(device).detach();
            avg_elbo, _,_ = model.forward(input_flat)
            losses['elbo'].append(avg_elbo.item())

    average_epoch_elbo = np.mean(losses['elbo'])
    return average_epoch_elbo,\
           (None if not losses['recon'] else np.mean(losses['recon']),
            None if not losses['kl'] else np.mean(losses['kl']))


def run_epoch(model, data, optimizer, device):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo, (train_rec, train_kl) = epoch_iter(model, traindata, optimizer, device)

    model.eval()
    val_elbo, (_, _) = epoch_iter(model, valdata, optimizer, device)

    return train_elbo, val_elbo, (train_rec, train_kl)


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def show(img, epoch_num):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.title(epoch_num)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    # fig1.savefig('0_gen_' + str(epoch_num) +'.png')


def manifold(ARGS):
    # code followed from :
    # https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder

    device = torch.cuda.current_device()
    model = VAE(z_dim=ARGS.zdim).to(device)
    model.load_state_dict(torch.load('results/z=2/39.model'))

    # Display a 2D manifold of the digits
    n = 20  # figure with 20x20 digits
    batch_size = 128
    figure = np.zeros((28 * n, 28 * n))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0, 1, n))
    grid_y = norm.ppf(np.linspace(0, 1, n))

    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = torch.tensor(np.tile(z_sample, batch_size)
                                    .reshape(batch_size, 2)).to(device).float().detach()
            x_decoded = model.decoder.forward(z_sample)
            digit = x_decoded[0].reshape(28, 28).detach().cpu()
            figure[i * 28: (i + 1) * 28,
            j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(figure, cmap='gnuplot2')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('0_manifold.png')


def main():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device(torch.cuda.current_device())

    data = bmnist(root='../data/')[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    train_recon_kl_curve = {'recon': [], 'kl':[]}
    for epoch in range(ARGS.epochs):
        train_elbo, val_elbo, (train_recon, train_kl) = run_epoch(model, data, optimizer, device)
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)

        train_recon_kl_curve['recon'].append(train_recon)
        train_recon_kl_curve['kl'].append(train_kl)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  plot samples from model during training.
        # --------------------------------------------------------------------
        if epoch in [0, 5, 10, 15, 20, 25, 30, 35, 39, 40]:
            sampled_ims, sampled_means = model.sample(100)
            sampled_grid = make_grid(sampled_ims.reshape(100, 1, 28, 28), nrow=10)
            show(sampled_grid, epoch)
            save_image(sampled_ims.reshape(100, 1, 28, 28), 'imgs_' + str(epoch) + '.png', nrow=10, normalize=True)
            save_image(sampled_means.reshape(100, 1, 28, 28), 'mean_' + str(epoch) + '.png', nrow=10, normalize=True)

    with open('39.pickle', 'wb') as handle:
        pickle.dump({'train_curve': train_curve, 'val_curve': val_curve,
                     'elbo_parts': train_recon_kl_curve}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    torch.save(model.state_dict(), str(epoch) + '.model')
    torch.save(optimizer.state_dict(), str(epoch) + '.optim')

    for k, key in enumerate(train_recon_kl_curve.keys()):
        plt.subplot(1, 2, k + 1)
        plt.plot(train_recon_kl_curve[key], label='train' + key)
        plt.title(key)

    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('0_recon_kl_loss.png')

    # --------------------------------------------------------------------
    #  plot plot the learned data manifold (if zdim == 2)
    # --------------------------------------------------------------------

    if ARGS.zdim == 2:
        manifold(ARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()
    ARGS.zdim = 2
    # main()
    manifold(ARGS)
