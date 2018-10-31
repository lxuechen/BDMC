from __future__ import print_function

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import log_normal, log_bernoulli, log_mean_exp


class VAE(nn.Module):

  def __init__(self, hps, seed=1):
    super(VAE, self).__init__()
    torch.manual_seed(seed)

    self.hps = hps

    self.dtype = torch.cuda.FloatTensor if hps.cuda else torch.FloatTensor
    self.init_layers()

    # if self.hps.has_flow:
    #     self.q_dist = Flow(self)
    #     if hps.cuda:
    #         self.q_dist.cuda()

  def init_layers(self):

    h_s = 500 if self.hps.wide_encoder else 200

    # MNIST
    self.fc1 = nn.Linear(784, h_s)
    self.fc2 = nn.Linear(h_s, h_s)
    self.fc3 = nn.Linear(h_s, self.hps.z_size * 2)

    if self.hps.large_encoder:
      self.fc_extra = nn.Linear(h_s, h_s)

    self.fc4 = nn.Linear(self.hps.z_size, 200)
    self.fc5 = nn.Linear(200, 200)
    self.fc6 = nn.Linear(200, 784)

  def sample(self, mu, logvar):

    bs = mu.size()[0]
    zs = self.hps.z_size

    eps = Variable(torch.FloatTensor(mu.size()).normal_().type(self.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)
    logqz = log_normal(z, mu, logvar)

    # if self.hps.has_flow:
    #     z, logprob = self.q_dist.forward(z)
    #     logqz += logprob

    zeros = Variable(torch.zeros(z.size()).type(self.dtype))
    logpz = log_normal(z, zeros, zeros)

    return z, logpz, logqz

  def encode(self, net):

    f = self.hps.act_func
    zs = self.hps.z_size

    net = f(self.fc1(net))
    net = f(self.fc2(net))

    if self.hps.large_encoder:
      net = f(self.fc_extra(net))

    net = self.fc3(net)

    mean, logvar = net[:, :zs], net[:, zs:]

    return mean, logvar

  def decode(self, net):

    f = self.hps.act_func

    net = f(self.fc4(net))
    net = f(self.fc5(net))
    net = self.fc6(net)

    x_logits = net

    return x_logits

  def forward(self, x, k=1, warmup_const=1.):

    x = x.repeat(k, 1)
    mu, logvar = self.encode(x)
    z, logpz, logqz = self.sample(mu, logvar)
    x_logits = self.decode(z)

    logpx = log_bernoulli(x_logits, x)
    elbo = logpx + logpz - warmup_const * logqz  # custom warmup

    # need correction for Tensor.repeat
    elbo = log_mean_exp(elbo.view(k, -1).transpose(0, 1))
    elbo = torch.mean(elbo)

    logpx = torch.mean(logpx)
    logpz = torch.mean(logpz)
    logqz = torch.mean(logqz)

    return elbo, logpx, logpz, logqz
