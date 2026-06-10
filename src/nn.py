import torch
import numpy as np
from itertools import product
from torch.nn.utils.parametrizations import orthogonal
from torch.func import vmap, jacfwd
import torch.nn as nn


def FourierKernel(t, T, n_kernel):
    mode = torch.arange(1, (n_kernel - 1) // 2 + 1)

    sin = torch.sin(2.0 * np.pi * mode * t / T)
    cos = torch.cos(2.0 * np.pi * mode * t / T)

    return torch.hstack((torch.tensor([1.0]), cos, sin))

def create_orbit(dim):
    def rotate(x, y):
        return y, dim - 1 - x

    def reflect(x, y):
        return dim - 1 - x, y

    translate = [(i, j) for i, j in product(*([range(dim)] * 2))]
    indices = -torch.ones((dim, dim, dim, dim))

    n = 0
    for a, b, c, d in product(*([range(dim)] * 4)):
        if indices[a, b, c, d] == - 1:
            for k in range(2):
                if k == 1:
                    ii, jj, xx, yy = *reflect(a, b), *reflect(c, d)
                else:
                    ii, jj, xx, yy = [a, b, c, d].copy()
                for l in range(dim * dim):
                    i, j = (ii + translate[l][0]) % dim, (jj + translate[l][1]) % dim
                    x, y = (xx + translate[l][0]) % dim, (yy + translate[l][1]) % dim
                    for m in range(4):
                        i, j, x, y = *rotate(i, j), *rotate(x, y)
                        indices[i, j, x, y] = n
            n += 1

    return indices.to(torch.long), n

#-----------phi^4 vector field architecture, following Gerdes et al---------------------#

# class PHIAnalytic(nn.Module):
#     def __init__(self, T, shape, n_kernel, n_kernel_bond, 
#                  n_basis, n_basis_bond, hutch=1):
#         super().__init__()
#         self.T = T
#         self.n_kernel = n_kernel

#         self.indices, self.n = create_orbit(shape[0])

#         self.W = nn.Parameter(torch.zeros((self.n, n_basis_bond, n_kernel_bond)))
#         self.k = nn.Parameter(orthogonal(nn.Linear(n_kernel, n_kernel_bond)).weight / n_kernel)
#         self.f = nn.Parameter(5 * torch.rand(n_basis - 1))
#         self.F = nn.Parameter(orthogonal(nn.Linear(n_basis, n_basis_bond)).weight / n_basis)

#         self.noise = torch.distributions.MultivariateNormal(torch.zeros(shape), torch.eye(shape[0]))
#         self.hutch = hutch

#     def forward(self, t, state, reverse = False, div = "hutch", eps = 0):
#         with torch.set_grad_enabled(True):
#             if div == "hutch":
#                 state[0].requires_grad_(True)
#             x_lin = torch.unsqueeze(state[0], dim=-1)
#             wf = x_lin * self.f
#             x = torch.concat((x_lin, torch.sin(wf)), dim=-1)
#             K = FourierKernel(t, self.T, self.n_kernel)

#             KK = self.k @ K
#             xx = torch.einsum('aijx, yx -> aijy', x, self.F)
#             ww = torch.einsum('ixy, y -> ix', self.W, KK)[self.indices]

#             dx = torch.einsum('aklx, ijklx -> aij', xx, ww)

#             if div == "None":
#                 return dx
#             elif div == "exact":
#                 y = torch.concat((torch.ones(x_lin.shape), torch.cos(wf) * self.f), dim=-1)
#                 yy = torch.einsum('aijx, yx -> aijy', y, self.F)
#                 dlogJ = torch.einsum('aijx, ijijx -> a', yy, ww)
#                 return dx, -dlogJ
#             elif div == "hutch":
#                 epsilon = self.noise.sample((self.hutch, state[0].shape[0]))
#                 jvp = torch.autograd.grad(dx, state[0], epsilon, allow_unused=True,create_graph=True,is_grads_batched=True)[0]
#                 dlogJ = torch.einsum('baij,baij->a', jvp, epsilon) / self.hutch
#                 return dx, -dlogJ

class PHIAnalyticUnbias(nn.Module):
    def __init__(self, T, shape, n_kernel, n_kernel_bond, n_basis, n_basis_bond,
                 hutch = 1, eps = 0):
        super().__init__()
        self.T = T
        self.n_kernel = n_kernel

        self.indices, self.n = create_orbit(shape[0])

        self.W = nn.Parameter(torch.zeros((self.n, n_basis_bond, n_kernel_bond, 2)))
        self.k = nn.Parameter(torch.stack([orthogonal(nn.Linear(n_kernel, n_kernel_bond)).weight / n_kernel,
                                           orthogonal(nn.Linear(n_kernel, n_kernel_bond)).weight / n_kernel], dim = 0))
        self.f = nn.Parameter(5 * torch.rand(2, n_basis - 1))
        self.F = nn.Parameter(torch.stack([orthogonal(nn.Linear(n_basis, n_basis_bond)).weight / n_basis, 
                                           orthogonal(nn.Linear(n_basis, n_basis_bond)).weight / n_basis], dim = 0))
        
        self.noise = torch.distributions.MultivariateNormal(torch.zeros(shape), torch.eye(shape[0]))
        self.hutch = hutch
        self.eps = eps

    def forward(self, t, state, reverse=False, div = "hutch"):
        with torch.set_grad_enabled(True):
            if div == "hutch":
                state[0].requires_grad_(True)
            x_lin = state[0]
            wf = torch.einsum('aij, bx -> baijx', x_lin, self.f)
            x = torch.concat((x_lin.repeat(2, 1, 1, 1).unsqueeze(dim=-1), torch.sin(wf)), dim=-1)
            K = FourierKernel(t, self.T, self.n_kernel)

            KK = self.k @ K
            xx = torch.einsum('baijx, byx -> baijy', x, self.F)
            ww = torch.einsum('ixyb, by -> ixb', self.W, KK)[self.indices]

            dx = torch.einsum('baklx, ijklxb -> baij', xx, ww)

            if div == "None":
                if reverse == True:
                    return dx[0] + self.eps * dx[1]
                else:
                    return dx[0] - self.eps * dx[1]
            elif div == "exact":
                y = torch.concat((torch.ones(x_lin.shape).repeat(2, 1, 1, 1).unsqueeze(dim=-1), torch.cos(wf) * self.f), dim=-1)
                yy = torch.einsum('aijx, yx -> aijy', y, self.F)
                dlogJ = torch.einsum('aijx, ijijx -> a', yy, ww)
                if reverse == True:
                    return dx[0] + self.eps * dx[1], -dlogJ[0] - self.eps * dlogJ[1]
                else:
                    return dx[0] - self.eps * dx[1], -dlogJ[0] + self.eps * dlogJ[1]
            elif div == "hutch":
                epsilon = self.noise.sample((self.hutch, 2, state[0].shape[0]))
                jvp = torch.autograd.grad(dx, state[0], epsilon, allow_unused=True,create_graph=True,is_grads_batched=True)[0]
                dlogJ = torch.einsum('bcaij,bcaij->ca', jvp, epsilon) / self.hutch
                if reverse == True:
                    return dx[0] + self.eps * dx[1], -dlogJ[0] - self.eps * dlogJ[1]
                else:
                    return dx[0] - self.eps * dx[1], -dlogJ[0] + self.eps * dlogJ[1]


#----------------GMM vectorf field architecture----------------------#

# class MLPGMM(nn.Module):
#     def __init__(self, dim, hidden, hutch=1):
#         super().__init__()
#         hidden = [dim + 1] + hidden + [dim]

#         layers = []
#         for i in range(len(hidden) - 2):
#             layers.append(nn.Linear(hidden[i], hidden[i + 1], bias=True))
#             layers.append(nn.GELU())
#         layers.append(nn.Linear(hidden[-2], hidden[-1], bias=True))

#         self.layers = nn.Sequential(*nn.ModuleList(layers))
#         self.hutch = hutch

#     def forward(self, T, state, reverse=False, div = "hutch", eps = 0):
#         def single_drift(x, t):
#             inp = torch.cat([x, t.unsqueeze(0)])
#             return self.layers(inp)
        
#         with torch.set_grad_enabled(True):
#             if div == "hutch":
#                 state[0].requires_grad_(True)
#             jvp_x = vmap(single_drift, in_dims=(0, 0), out_dims=(0), 
#                          randomness='different')(state[0], T.repeat(len(state[0]))).squeeze()
        
#             if div == "None":
#                 return jvp_x
#             elif div == "exact":
#                 dlogJ = vmap(jacfwd(single_drift, argnums=0), out_dims=(0), 
#                              randomness='different')(state[0], T.repeat(len(state[0]))).squeeze()
#                 dlogJ = torch.einsum('ijj -> i', dlogJ)
#                 return jvp_x, -dlogJ
#             elif div == "hutch":
#                 epsilon = torch.randn((self.hutch, ) + state[0].shape)
#                 jvp = torch.autograd.grad(jvp_x, state[0], epsilon, allow_unused=True,create_graph=True,is_grads_batched=True)[0]
#                 dlogJ = torch.einsum('ba...,ba...->a', jvp, epsilon) / self.hutch
#                 return jvp_x, -dlogJ

class MLPGMMUnbias(nn.Module):
    def __init__(self, dim, hidden, hutch=1, eps=0):
        super().__init__()
        hidden = [dim + 1] + hidden + [2 * dim]

        layers = []
        for i in range(len(hidden) - 2):
            layers.append(nn.Linear(hidden[i], hidden[i + 1], bias=True))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden[-2], hidden[-1], bias=True))

        self.layers = nn.Sequential(*nn.ModuleList(layers))
        self.dim = dim
        self.eps = eps
        self.hutch = hutch

    def forward(self, T, state, reverse=False, div = "hutch", eps = 0):
        def single_drift(x, t):
            inp = torch.cat([x, t.unsqueeze(0)])
            return self.layers(inp).reshape(2, self.dim)
        
        with torch.set_grad_enabled(True):
            if div == "hutch":
                state[0].requires_grad_(True)
            jvp_x = vmap(single_drift, in_dims=(0, 0), out_dims=(0), 
                            randomness='different')(state[0], T.repeat(len(state[0]))).squeeze()
                
            if div == "None":
                if reverse == True:
                    return jvp_x[:,0] + self.eps * jvp_x[:,1]
                else:
                    return jvp_x[:,0] - self.eps * jvp_x[:,1]
            elif div == "exact":
                dlogJ = vmap(jacfwd(single_drift, argnums=0), out_dims=(0), 
                             randomness='different')(state[0], T.repeat(len(state[0]))).squeeze()
                dlogJ = torch.einsum('ijkk -> ji', dlogJ)
                if reverse == True:
                    return jvp_x[:,0] + self.eps * jvp_x[:,1], -dlogJ[0] - self.eps * dlogJ[1]
                else:
                    return jvp_x[:,0] - self.eps * jvp_x[:,1], -dlogJ[0] + self.eps * dlogJ[1]
            elif div == "hutch":
                epsilon = torch.randn((self.hutch, 2) + state[0].shape)
                jvp = torch.autograd.grad(jvp_x, state[0], epsilon, allow_unused=True,create_graph=True,is_grads_batched=True)[0]
                dlogJ = torch.einsum('bca...,bca...->ca', jvp, epsilon) / self.hutch
                if reverse == True:
                    return jvp_x[:,0] + self.eps * jvp_x[:,1], -dlogJ[0] - self.eps * dlogJ[1]
                else:
                    return jvp_x[:,0] - self.eps * jvp_x[:,1], -dlogJ[0] + self.eps * dlogJ[1]