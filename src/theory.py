import torch
import torch.distributions as D
import torch.nn.functional as f
from torch.distributions import MixtureSameFamily

#--------------------------phi^4----------------------------------#

class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size, ))
        return x.reshape(batch_size, *self.shape)

class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam
    def __call__(self, cfgs):
        #potential term
        action_density = self.M2 * cfgs**2 + self.lam * cfgs**4
        #kinetic term (discrete laplacian)
        Nd = len(cfgs.shape) - 1
        dims = range(1, Nd+1)
        for mu in dims:
            action_density += 2 * cfgs**2
            action_density -= cfgs * torch.roll(cfgs, -1, mu)
            action_density -= cfgs * torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim = tuple(dims))
    #log_prob is -action
    def log_prob(self, cfgs):
        action_density = self.M2 * cfgs**2 + self.lam * cfgs**4
        Nd = len(cfgs.shape) - 1
        dims = range(1, Nd+1)
        for mu in dims:
            action_density += 2 * cfgs**2
            action_density -= cfgs * torch.roll(cfgs, -1, mu)
            action_density -= cfgs * torch.roll(cfgs, 1, mu)
        return -torch.sum(action_density, dim = tuple(dims))

#--------------------------GMM----------------------------------#

def create_gmm_normal(means):
    dim = means.shape[1]
    n_mixes = means.shape[0]
    log_var_scaling = 1.0

    log_var = torch.ones((n_mixes, dim)) * log_var_scaling
    scale_trils = torch.diag_embed(f.softplus(log_var))

    mix = D.Categorical(torch.ones(n_mixes,))
    com = D.MultivariateNormal(means, scale_trils, validate_args=False)
    gmm = MixtureSameFamily(mix, com, validate_args=False)

    return gmm

def sample_cov(ndim, nwell, rng):
    sigmas = torch.zeros((nwell, ndim, ndim))
    sigmas_diag = 0.01 + torch.abs(torch.normal(0.0, 0.25, size=(ndim,), generator=rng))
    A = torch.normal(0, 1, (ndim, ndim), generator=rng)
    Q, R = torch.linalg.qr(A)
    sigmas[1] = Q @ torch.diag(sigmas_diag) @ Q.T

    sigmas_diag = 0.01 + torch.abs(torch.normal(0.0, 0.25, size=(ndim,), generator=rng))
    sigmas[0] = torch.diag(sigmas_diag)

    return sigmas

def create_gmm_ndim(n = 1000, rng=None):
    means = torch.zeros((2, n))
    means[0, 0] = 2.0
    means[1, 0] = -2.0
    sigmas = sample_cov(n, 2, rng)
    mix = D.Categorical(torch.tensor([0.25, 0.75]))
    com = D.MultivariateNormal(means, sigmas, validate_args=False)
    gmm = MixtureSameFamily(mix, com, validate_args=False)

    return gmm