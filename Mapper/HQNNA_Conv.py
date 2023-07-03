import torch
import numpy as np
import math
random_seed = 1
torch.manual_seed(random_seed)

# ! HQNNA Convolution mapping file 
# ! HQNNA to perform dot product operation between two 16-bit precision numbers, divides them into 4-bit slices and performs the dot product operation on each slice. Later 
# ! each slice result is shifted and added to get the final result.
# ! HQNNA defines b=4 DPEs to support the bit slices 
# ! To mimic these execution, instead of slicing the number I follow the below approach
# ! Allocate each dot product 4 DPEs, first DPE contains the original value of weight and rest three DPEs are filled with zeros
# ! For achieving slicing at the input I run dummy loops to mimic the slicing operation of activations


C = 4
D = 4
K = 4
N = 4 # size of DPE
M = 4 # Number of DPEs in a DPU
Y = 10 # Number of DPUs
I = torch.randn(C,K)
W = torch.randn(K,D)
O = torch.zeros(C,D)
required_precision = 1
sup_act_precision = 2
B = 4  # Here B is the number of DPEs to support different bit shifted numbers
#! Intra DPU Sharing
print("Input ", I)
print("Weight ", W)
num_of_bit_slice = math.ceil(required_precision/sup_act_precision)

assert (M%B == 0), "Total DPE count should be divisible by BitSlices"
for bit_slice in range(num_of_bit_slice):
  O = torch.zeros(C,D)
  for c in range(0, C, Y):
      for d in range(0, D, math.ceil(M/B)):
          for k in range(0, K, N):
              i_slice = I[c: min(c+Y,C), k : min(k + N, K)]
              w_slice = W[k : min(k + N, K), d:min(d+math.ceil(M/B),D)]
              print('W Slice ',w_slice)
              w_slice = w_slice.T
              zero_index_list = torch.tensor([i for i in range(M) if i % B == 0])
              zero_index_list = zero_index_list[0:w_slice.shape[0]]
              # print("Zero Index :",zero_index_list)
              # print('I Slice',i_slice)
            
              size = (M,min(k + N, K)-k)
              print('Size ', size)
              for dpu_idx in range(min(c+Y,C)-c):
                print("DPU Number", dpu_idx)
                dpu_i_slice = i_slice[dpu_idx,:]
                dpu_i_slice = dpu_i_slice.T.repeat(M,1)
                dpu_w_slice = w_slice
                dpu_w_slice = torch.zeros(size)

                dpu_w_slice.index_add_(0, zero_index_list, w_slice)
                # dpu_w_slice = torch.cat([dpu_w_slice, torch.zeros(B-1, dpu_w_slice.shape[1], dtype=dpu_w_slice.dtype)], dim=0)
                # print(dpu_w_slice)
                # dpu_w_slice = dpu_w_slice.repeat(1, 1 + B).reshape(-1, dpu_w_slice.shape[1])
                print("DPU I Slice ", dpu_i_slice)
                print("DPU W Slice ", dpu_w_slice)
                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
                print("Psum DPU", psum_dpu)
                cluster_psum_dpu = psum_dpu.reshape(math.ceil(M/B),B)
                print("Cluster DPU", cluster_psum_dpu)
                reduced_psum_dpu =  cluster_psum_dpu.sum(dim=1)
                print("Reduced PSum DPU", reduced_psum_dpu)
                reduced_psum_dpu = reduced_psum_dpu[0:w_slice.shape[0]]
                O[c+dpu_idx,d:d+math.ceil(M/B)] = reduced_psum_dpu+O[c+dpu_idx,d:d+math.ceil(M/B)]
                print(O)
print(O)
print(I @ W)
print(torch.allclose(O, I @W))