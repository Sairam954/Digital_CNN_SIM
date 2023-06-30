import torch
import numpy as np
random_seed = 1
torch.manual_seed(random_seed)

C = 10
D = 10
K = 3
N = 2 # size of DPE
M = 2 # Number of DPEs in a DPU
Y = 2 # Number of DPUs
I = torch.randn(C,K)
W = torch.randn(K,D)
O = torch.zeros(C,D)

#! Intra DPU Sharing 
print("Input ", I)
print("Weight ", W)

for c in range(0, C, M):
    for d in range(0, D, Y):
        for k in range(0, K, N):
            i_slice = I[c: min(c+Y,C), k : min(k + N, K)]
            w_slice = W[k : min(k + N, K), d:min(d+M,D)]
            w_slice = w_slice.T
            for dpu_idx in range(min(d+Y,D)-d):
              dpu_i_slice = i_slice[dpu_idx,:]
              dpu_i_slice = dpu_i_slice.T.repeat(min(d+M,D)-d,1)
              dpu_w_slice = w_slice
              psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
              O[c+dpu_idx,d:d+M] = psum_dpu+O[c+dpu_idx,d:d+M]
print(O)
print(I @ W)
print(torch.allclose(O, I @W))