import torchdiffeq
import torch
from integrator import sdeint, rk4int, fp4int
from utils import grab, calc_dkl, compute_ess

def eval_step(model, action, prior, times, mode, theory,
              eps = 0.05, batch_size = 2000, num_fp_noise = 10, Nboot = 2000):
    with torch.no_grad():
        x = prior.sample_n(batch_size)
        logq = prior.log_prob(x)

        if mode == 'unbias':
            noises = torch.randn(len(times), *x.shape)
            x, logq = sdeint(model, x, logq, eps, times, noises)
        elif mode == 'fp':
            noises = torch.randn(len(times), num_fp_noise, *x.shape)
            x, logq = fp4int(model, x, logq, times, noises)
        elif mode == 'hutch':
            x, logq = torchdiffeq.odeint(model, (x, logq), times, method='rk4')
            x, logq = x[-1], logq[-1]
        else:
            x, logq = rk4int(model, x, logq, times, div='exact')
        
        logp = action.log_prob(x)

        boots = torch.mean(logp[torch.randint(len(x), size=(Nboot, len(x)))], -1)
        logp_mean = boots.mean()
        logp_stdr = boots.std()
        
        logw = logp - logq
        boots = torch.mean(-logw[torch.randint(len(x), size=(Nboot, len(x)))], -1)
        loss_mean = boots.mean()
        loss_stdr = boots.std()
        
        boots = torch.logsumexp(logw[torch.randint(len(x), size=(Nboot, len(x)))], -1)
        part_mean = (torch.exp(boots) / len(logw)).mean()
        part_stdr = (torch.exp(boots) / len(logw)).std()
        
        boots = torch.logsumexp(logw[torch.randint(len(x), size=(Nboot, len(x)))], -1)
        free_mean = (boots - torch.log(torch.tensor(len(logw)))).mean()
        free_stdr = (boots - torch.log(torch.tensor(len(logw)))).std()
        
        logww = logw[torch.randint(len(x), size=(Nboot, len(x)))]
        log_ess = 2 * torch.logsumexp(logww, dim=-1) - torch.logsumexp(2 * logww, dim=-1)
        ess_per_cfg = torch.exp(log_ess) / len(x)
        ess_mean = ess_per_cfg.mean()
        ess_stdr = ess_per_cfg.std()

        results = [logp_mean.item(), logp_stdr.item(), loss_mean.item(), loss_stdr.item(),
               part_mean.item(), part_stdr.item(), free_mean.item(), free_stdr.item(),
               ess_mean.item(), ess_stdr.item()]

        if theory == "phi":
            magn = torch.mean(x, (-1, -2))
            boots = torch.mean(magn[torch.randint(len(x), size=(Nboot, len(x)))], -1)
            magn_mean = boots.mean()
            magn_stdr = boots.std()
            results += [magn_mean.item(), magn_stdr.item()]

    return results