import os
import torch
import numpy as np

def calc_dkl(logp, logq):
    return (logq - logp).mean()   #reverse KL, assuming samples from q

def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg

def bootstrap(x:np.ndarray, *, Nboot = 1000, binsize = 4) -> tuple[float, float]:
    boots = []

    # Divide the samples into sub-samples of size = binsize
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(np.mean(x[np.random.randint(len(x), size=len(x))], axis=(0,1)))
    return np.mean(boots), np.std(boots)

def grab(var):
    return var.detach().cpu().numpy()

def join_paths(base, sub):
    return os.path.normpath(os.path.join(base, sub))