import torchdiffeq
import torch
from integrator import sdeint_checkpoint, rk4int_checkpoint, fp4int_checkpoint
from utils import grab, calc_dkl, compute_ess

def train_step(model, action, prior, optimizer, metrics, times, mode,
               eps = 0.05, batch_size = 256, num_fp_noise = 10, num_checkpoint = 50):
    optimizer.zero_grad()

    x = prior.sample_n(batch_size)
    logq = prior.log_prob(x)

    if mode == 'unbias':
        noises = torch.randn(len(times), *x.shape)
        x, logq = sdeint_checkpoint(model, x, logq, eps, times, noises, num_checkpoint)
    elif mode == 'fp':
        noises = torch.randn(len(times), num_fp_noise, *x.shape)
        x, logq = fp4int_checkpoint(model, x, logq, times, noises, num_checkpoint)
    elif mode == 'hutch':
        x, logq = torchdiffeq.odeint_adjoint(model, (x, logq), times, method='rk4')
        x, logq = x[-1], logq[-1]
    else:
        x, logq = rk4int_checkpoint(model, x, logq, times, 'exact', num_checkpoint)

    logp = action.log_prob(x)
    loss = calc_dkl(logp, logq)
    loss.backward()

    optimizer.step()

    metrics['loss'].append(grab(loss))
    metrics['logp'].append(grab(logp))
    metrics['logq'].append(grab(logq))
    metrics['ess'].append(grab(compute_ess(logp, logq)))