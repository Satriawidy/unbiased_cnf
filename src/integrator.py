import torch
from torch.func import jvp
from torch.utils.checkpoint import checkpoint

def sdeint(model, x, logq, eps, times, noises):
    for noise, t0, t1 in zip(noises[:-1], times[:-1], times[1:]):
        dt = t1 - t0
        drift = model.forward(t0, (x, logq), reverse=False, div="None", eps=eps)
        P = noise * torch.sqrt(2 * eps * torch.abs(dt))
        x = x + drift * dt + P
        M = (model.forward(t0, (x, logq), reverse=True, div="None",  eps=eps) - drift) * dt - P
        Rp = torch.einsum('i..., i... -> i', P, P) / (4 * eps * torch.abs(dt))
        Rm = torch.einsum('i..., i... -> i', M, M) / (4 * eps * torch.abs(dt))
        logq = logq - Rp + Rm
    return x, logq

def rk4int(model, x, logq, times, div="None"):
    if div == "None":
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            k1 = model.forward(t0, (x, logq), div=div)
            k2 = model.forward(t0 + dt / 3, (x + k1 * dt / 3, logq), div=div)
            k3 = model.forward(t0 + 2 * dt / 3, (x - k1 * dt / 3 + k2 * dt, logq), div=div)
            k4 = model.forward(t0 + dt, (x + k1 * dt - k2 * dt + k3 * dt, logq), div=div)
            x = x + dt * (k1 + 3 * k2 + 3 * k3 + k4) / 8
        return x
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1 - t0
        k1, d1 = model.forward(t0, (x, logq), div=div)
        k2, d2 = model.forward(t0 + dt / 3, (x + k1 * dt / 3, logq), div=div)
        k3, d3 = model.forward(t0 + 2 * dt / 3, (x - k1 * dt / 3 + k2 * dt, logq), div=div)
        k4, d4 = model.forward(t0 + dt, (x + k1 * dt - k2 * dt + k3 * dt, logq), div=div)
        x = x + dt * (k1 + 3 * k2 + 3 * k3 + k4) / 8
        logq = logq + dt * (d1 + 3 * d2 + 3 * d3 + d4) / 8
    return x, logq

def fp4int(model, x, logq, times, noises):
    vol = 1
    for j in x.shape[1:]:
        vol *= j
    for noise, t0, t1 in zip(noises, times[:-1], times[1:]):
        x = rk4int(model, x, logq, [t0, t1])
        noisy = noise / torch.sum(noise**2, tuple(range(2, len(noise.shape))), keepdims = True).sqrt()
        def func(x):
            return rk4int(model, x, logq, [t1, t0])

        jvps = torch.stack([jvp(func, (x, ), (noisy[i], ))[1] for i in range(len(noisy))], 0)
        logq = logq - torch.logsumexp(-vol * torch.log(torch.einsum('ba..., ba... -> ba', 
                                                                                 jvps, jvps).sqrt()), 0)
        logq = logq + torch.log(torch.tensor(len(noisy)))
    return x, logq

def sdeint_checkpoint(model, x, logq, eps, times, noises, n):
    m, r = 0, len(times)
    while m < r - 1:
        x, logq = checkpoint(sdeint, model, x, logq, eps, times[m:m+n+1],
                             noises[m:m+n+1], use_reentrant=False)
        m = m + n
    return x, logq

def rk4int_checkpoint(model, x, logq, times, div, n):
    m, r = 0, len(times)
    while m < r - 1:
        x, logq = checkpoint(rk4int, model, x, logq, times[m:m+n+1],
                             div, use_reentrant=False)
        m = m + n
    return x, logq

def fp4int_checkpoint(model, x, logq, times, noises, n):
    m, r = 0, len(times)
    while m < r - 1:
        x, logq = checkpoint(fp4int, model, x, logq, times[m:m+n+1],
                             noises[m:m+n+1], use_reentrant=False)
        m = m + n
    return x, logq