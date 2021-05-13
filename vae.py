import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class VAE(nn.Module):

  def __init__(self, latent_dim=50, act_fn=F.elu):
    super(VAE, self).__init__()

    self.latent_dim = latent_dim
    self.act_fn = act_fn

    self.fc1 = nn.Linear(784, 200)
    self.fc2 = nn.Linear(200, 200)
    self.fc3 = nn.Linear(200, latent_dim * 2)

    self.fc4 = nn.Linear(latent_dim, 200)
    self.fc5 = nn.Linear(200, 200)
    self.fc6 = nn.Linear(200, 784)

  def sample(self, mu, logvar):
    eps = torch.randn(mu.size()).cuda()
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)
    logqz = utils.log_normal(z, mu, logvar)

    zeros = torch.zeros(z.size()).cuda()
    logpz = utils.log_normal(z, zeros, zeros)

    return z, logpz, logqz

  def encode(self, net):
    net = self.act_fn(self.fc1(net))
    net = self.act_fn(self.fc2(net))
    net = self.fc3(net)

    return net[:, :zs], net[:, zs:]

  def decode(self, net):
    net = self.act_fn(self.fc4(net))
    net = self.act_fn(self.fc5(net))
    return self.fc6(net)

  def forward(self, x, k=1, warmup_const=1.):
    x = x.repeat(k, 1)
    mu, logvar = self.encode(x)
    z, logpz, logqz = self.sample(mu, logvar)
    x_logits = self.decode(z)

    logpx = utils.log_bernoulli(x_logits, x)
    elbo = logpx + logpz - warmup_const * logqz

    # need correction for Tensor.repeat
    elbo = utils.log_mean_exp(elbo.view(k, -1).transpose(0, 1))
    elbo = torch.mean(elbo)

    logpx = torch.mean(logpx)
    logpz = torch.mean(logpz)
    logqz = torch.mean(logqz)

    return elbo, logpx, logpz, logqz
