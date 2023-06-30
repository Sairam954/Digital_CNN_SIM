import torch
import numpy as np
random_seed = 1
torch.manual_seed(random_seed)

C = 10
D = 10
K = 10
N = 2 # size of DPE
M = 2 # Number of DPEs in a DPU
Y = 2 # Number of DPUs
I = torch.randn(C,K)
W = torch.randn(K,D)
O = torch.zeros(C,D)

#! Intra DPU Sharing 


for c in range(0,C,M):
    for k in range(0, K, N):
        i_slice = I[c: min(c+M,C), k : min(k + N, K)]
        for d in range(0, D, Y):
            w_slice = W[k : min(k + N, K), d:min(d+Y,D)]
            for dpu_idx in range(min(d+Y,D)-d):
              dpu_w_slice = w_slice[:,dpu_idx]
              dpu_w_slice = dpu_w_slice.T.repeat(min(c+M,C)-c,1)
              psum_dpu = torch.einsum('ij,ij->i', i_slice, dpu_w_slice)
              O[c:c+M,d+dpu_idx] = psum_dpu+O[c:c+M,d+dpu_idx]
print(O)
print(I @ W)
print(torch.allclose(O, I @W))