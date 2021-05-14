# BDMC

PyTorch implementation of Bidirectional Monte Carlo.

## Requirements

* `python3`
* `numpy`
* `pytorch>=0.4.1`
* `tqdm`

## What is Bidirectional Monte Carlo (BDMC)?

BDMC is a method of accurately sandwiching the log marginal likelihood (ML). It is mainly used to evaluate the quality
of log-ML estimators [1]. The method achieves this by obtaining a lower bound with the usual Annealed Importance
Sampling (AIS) [2], and an upper bound with Reversed AIS from an exact posterior sample. Since the upper bound requires
an *exact* sample from the posterior, the method is only strictly valid on simulated data. However, the results obtained
on simulated data can help verify the performance of log-ML estimators. Conditioned upon the assumption that the real
data does not differ too much from the simulated data, the evaluation of the log-ML estimator on simulated data could be
informative of the performance on real data.

The given implementation performs evaluation on a variational autoencoder (VAE) trained on MNIST.

## To run

There is a pretrained VAE model (on MNIST) in the `checkpoints` folder. Executing the command

```bash
python bdmc.py \
  --latent-dim 50 \
  --batch-size 512 \
  --n-batch 2 \
  --chain-length 10000 \
  --iwae-samples 10 \
  --ckpt-path ./checkpoints/model.pth
```

will start the forward and backwards chain of BDMC based on the model loaded from the pretrained checkpoint.

## Implementation Details

To avoid OOM when using extravagantly long AIS chains, strategically using `tensor.requires_grad_()` is required. An
equivalent TensorFlow graph-mode implementation ideally would require using `tf.while_loop` or abstracting only part of
the graph (calling `session.run` multiple times during a chain).

## Others

Since BDMC relies on AIS, and AIS (potentially) relies on Hamiltonian Monte Carlo (HMC) [3], the repo also contains such
relevant code.

## References

[1] Grosse, Roger B., Zoubin Ghahramani, and Ryan P. Adams. "Sandwiching the marginal likelihood using bidirectional
Monte Carlo." arXiv preprint arXiv:1511.02543 (2015).

[2] Neal, Radford M. "Annealed importance sampling." Statistics and computing 11.2 (2001): 125-139.

[3] Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo 2.11 (2011).
