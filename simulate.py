import torch
from torch.distributions import Bernoulli


def simulate_data(model, batch_size=10, n_batch=1, device=None):
    """Simulate data from the VAE model. Sample from the
    joint distribution p(z)p(x|z). This is equivalent to
    sampling from p(x)p(z|x), i.e. z is from the posterior.

    Bidirectional Monte Carlo only works on simulated data,
    where we could obtain exact posterior samples.

    Args:
        model: VAE model for simulation
        batch_size: batch size for simulated data
        n_batch: number of batches
        device (torch.device): device to run all computation on

    Returns:
        iterator that loops over batches of torch Tensor pair x, z
    """
    batches = []
    for i in range(n_batch):
        # assume prior for VAE is unit Gaussian
        z = torch.randn(batch_size, model.latent_dim).to(device)
        x_logits = model.decode(z)
        if isinstance(x_logits, tuple):
            x_logits = x_logits[0]
        x_bernoulli_dist = Bernoulli(probs=x_logits.sigmoid())
        x = x_bernoulli_dist.sample().data

        paired_batch = (x, z)
        batches.append(paired_batch)

    return iter(batches)
