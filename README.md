# BDMC
PyTorch implementation of BDMC

## Requirements
* `python3`
* `numpy`
* `pytorch`

## What is Bidirectional Monte Carlo (BDMC)?
BDMC is method of accurately sandwiching the log marginal likelihood (ML) mainly used to evaluate the quality of log-ML estimators [1]. The method achieves this by obtaining an lower bound with the usual Annealed Importance Sampling (AIS) [2], and an upper bound with Reversed AIS from an exact posterior sample. Since the upper bound requires an *exact* sample from the posterior, the method is only strictly valid on simulated data. However, the results obtained on simulated data is still valid if it is very similar to real data. 

The given implementation performs evaluation on a variational autoencoder (VAE) trained on MNIST. 

## Other Stuff
Since BDMC relies on AIS, and AIS relies on Hamiltonian Monte Carlo (HMC), the repo also contains such relevant code. 

## References
[1] Grosse, Roger B., Zoubin Ghahramani, and Ryan P. Adams. "Sandwiching the marginal likelihood using bidirectional Monte Carlo." arXiv preprint arXiv:1511.02543 (2015).
[2] Neal, Radford M. "Annealed importance sampling." Statistics and computing 11.2 (2001): 125-139.
