import torch
import numpy as np
import math
from ADC import ADC
from ADC.ADC_16bit import ADC_16b
from DAC import DAC
from DAC.DAC_4bit import DAC_4b

from MRR_DPE import MRR_DPE
from PD import PD
from ReductionNetwork import RN
from SOA import SOA
from VoltageAdder import VoltageAdder
import pandas as pd

random_seed = 1
torch.manual_seed(random_seed)

# ! HQNNA Convolution mapping file 
# ! HQNNA to perform dot product operation between two 16-bit precision numbers, divides them into 4-bit slices and performs the dot product operation on each slice. Later 
# ! each slice result is shifted and added to get the final result.
# ! HQNNA defines b=4 DPEs to support the bit slices 
# ! To mimic these execution, instead of slicing the number I follow the below approach
# ! Allocate each dot product 4 DPEs, first DPE contains the original value of weight and rest three DPEs are filled with zeros
# ! For achieving slicing at the input I run dummy loops to mimic the slicing operation of activations


# cacha latency parameters
cacheMissRatioDf = pd.read_csv('C:\\Users\\SSR226\\Desktop\\MRRCNNSIM\\CacheUtils\\Miss_Ratio_Analysis1.csv')
cacheParameters = pd.read_csv('C:\\Users\\SSR226\\Desktop\\DataflowTesting\\CacheUtils\\Cache_Parameters.csv')
l1_latency = cacheParameters[cacheParameters['cache']=='l1']
l2_latency = cacheParameters[cacheParameters['cache']=='l2']

# MRR DPE Latencies
dac_latency = 0
prop_latency = 0
input_actuation_latency = 0
weight_actuation_latency = 0
dpe_latency = 0 # sum of prop, input_actuation and weight_actuation latency
soa_latency = 0
adc_latency = 0
pd_latency = 0

# cache access latencies 
psum_access_latency = 0
input_access_latency = 0
weight_access_latency  = 0 
output_access_latency = 0


# Psum reduction latency at RN 
psum_reduction_latency = 0 

# access counter for cache
psum_access_counter = 0
input_access_counter = 0
weight_access_counter = 0
output_access_counter = 0

# different energy parameters computed for GeMM execution
# MRR DPE Energy
weight_actuation_energy = 0
input_actuation_energy = 0
soa_energy = 0
pd_energy = 0
dac_energy = 0
adc_energy = 0

# cache access energy
weight_access_energy = 0
input_access_energy = 0
psum_access_energy = 0
partial_sum_reduction_energy = 0
output_access_energy = 0

# Mrr Utilization
used_mrr_counter = 0
unused_mrr_counter = 0

# Reduction Folds counter: To know how many times temporal reduction is used by a DPU        
folds_counter = 0

# storing metrics
latency_dict = {}
access_dict = {}
energy_dict = {}
soa_energy = {}




C = 4
D = 4
K = 4
N = 4 # size of DPE
M = 4 # Number of DPEs in a DPU
Y = 10 # Number of DPUs
I = torch.randn(C,K)
W = torch.randn(K,D)
O = torch.zeros(C,D)
required_precision = 8
sup_act_precision = 4
B = 4  # Here B is the number of DPEs to support different bit shifted numbers
#! Intra DPU Sharing
print("Input ", I)
print("Weight ", W)
num_of_bit_slice = math.ceil(required_precision/sup_act_precision)
reduction_network_type = "S_Tree"
data_rate = 1
miss_ratio = cacheMissRatioDf.loc[(cacheMissRatioDf['C']==C) & (cacheMissRatioDf['D']==D) & (cacheMissRatioDf['K']==K) & (cacheMissRatioDf['dataflow']== 'OS')]
# components 
dpe_obj = MRR_DPE(N,data_rate)
rn_obj = RN(reduction_network_type)
dac_obj = DAC_4b()
adc_obj = ADC_16b()
soa_obj = SOA()
pd_obj = PD()

ps_to_sec = 1e-12
ns_to_sec = 1e-9
us_to_sec = 1e-6

fJ_to_J = 1e-15
pJ_to_J = 1e-12
nJ_to_J = 1e-9
mW_to_W = 1e-3


assert (M%B == 0), "Total DPE count should be divisible by BitSlices"
for bit_slice in range(num_of_bit_slice):
  O = torch.zeros(C,D)
  for c in range(0, C, Y):
      for d in range(0, D, math.ceil(M/B)):
          temp_partial_sum_counter = 0
          for k in range(0, K, N):
              i_slice = I[c: min(c+Y,C), k : min(k + N, K)]
              w_slice = W[k : min(k + N, K), d:min(d+math.ceil(M/B),D)]
              
              # latency parameters calculations
              dac_latency +=  dac_obj.latency*ns_to_sec
              weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec
              input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
              prop_latency +=  dpe_obj.get_prop_latency()
              soa_latency += soa_obj.latency*ns_to_sec
              pd_latency += pd_obj.latency*ps_to_sec
              adc_latency += adc_obj.latency*ns_to_sec
              
              # energy parameters calculations
              
              
              print('W Slice ',w_slice)
              w_slice = w_slice.T
              zero_index_list = torch.tensor([i for i in range(M) if i % B == 0])
              zero_index_list = zero_index_list[0:w_slice.shape[0]]
              # print("Zero Index :",zero_index_list)
              # print('I Slice',i_slice)
              
              
            
              size = (M,min(k + N, K)-k)
              print('Size ', size)
              for dpu_idx in range(min(c+Y,C)-c):
                dpu_i_slice = i_slice[dpu_idx,:]
                
                
                dac_energy += dac_obj.energy*pJ_to_J*torch.numel(i_slice)
                dac_energy += dac_obj.energy*pJ_to_J*torch.numel(w_slice)*B*math.ceil(required_precision/16)
                input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                
                dpu_i_slice = dpu_i_slice.T.repeat(M,1)
                dpu_w_slice = w_slice
                dpu_w_slice = torch.zeros(size)
                
                
                dpu_w_slice.index_add_(0, zero_index_list, w_slice)
                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
                
                soa_energy += soa_obj.energy*pJ_to_J*w_slice.shape[0]*B*math.ceil(required_precision/16)
                pd_energy += pd_obj.energy*fJ_to_J*w_slice.shape[0]*B*math.ceil(required_precision/16)
                
                
                cluster_psum_dpu = psum_dpu.reshape(math.ceil(M/B),B)
                reduced_psum_dpu =  cluster_psum_dpu.sum(dim=1)
                reduced_psum_dpu = reduced_psum_dpu[0:w_slice.shape[0]]
                adc_energy += adc_obj.energy*pJ_to_J*torch.numel(reduced_psum_dpu)
                O[c+dpu_idx,d:d+math.ceil(M/B)] = reduced_psum_dpu+O[c+dpu_idx,d:d+math.ceil(M/B)]
                
                temp_partial_sum_counter += torch.numel(reduced_psum_dpu)
                psum_access_latency += 2*torch.numel(psum_dpu)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                 
          psum_reduction_latency += rn_obj.get_reduction_latency(temp_partial_sum_counter,1)          
          partial_sum_reduction_energy += rn_obj.get_reduction_latency(temp_partial_sum_counter,1)*rn_obj.power

print(O)
print(I @ W)
print(torch.allclose(O, I @W))