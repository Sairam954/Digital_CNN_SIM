import torch
import numpy as np
import torch.nn.functional as F
import math
random_seed = 1
torch.manual_seed(random_seed)

C = 4
D = 4
K = 3
N = 2 # size of DPE
M = 2 # Number of DPEs in a DPU
Y = 2 # Number of DPUs
I = torch.randn(C,K)
W = torch.randn(K,D)
O = torch.zeros(C,D)
L = 2 # Number of Clusters in a DPU


Z = math.ceil(M/L) # no of dpes in a single cluster


print('Input I', I)
print('Weight W', W)
psum_access_counter = 0
for c in range(0, C, Y):
    for d in range(0, D, L):
        for k in range(0, K, N*Z):
            i_slice = I[c: min(c+Y,C), k : min(k + N*Z, K)]
            i_temp = i_slice
            w_slice = W[k : min(k + N*Z, K), d:min(d+L,D)]
            w_temp = w_slice.T
            for dpu_idx in range(min(c+Y,C)-c):
                dpu_w_slice = w_temp

                dpu_i_slice = i_temp[dpu_idx,:]
                dpu_i_slice = dpu_i_slice.repeat(w_temp.shape[0],1)

                dpu_w_slice = F.pad(dpu_w_slice, (0, max(0, N*Z-dpu_w_slice.shape[1])), mode='constant', value=0)
                dpu_i_slice = F.pad(dpu_i_slice, (0, max(0, N*Z-dpu_i_slice.shape[1])), mode='constant', value=0)


                dpu_i_slice = dpu_i_slice.reshape(dpu_w_slice.shape[0]*Z,N)
                dpu_w_slice = dpu_w_slice.reshape(dpu_w_slice.shape[0]*Z,N)
                
                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
                cluster_psum_dpu = psum_dpu.unfold(0, Z, Z)
                reduced_psum_dpu =  cluster_psum_dpu.sum(dim=1)
                psum_access_counter += 2*L
                O[c+dpu_idx,d:min(d+L,D)] = reduced_psum_dpu+ O[c+dpu_idx,d:min(d+L,D)]

print(O)
print(I @W)
print(torch.allclose(O, I @W))