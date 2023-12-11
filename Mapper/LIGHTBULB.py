import torch
import numpy as np
import math
from ADC import ADC, ADC_8bit
from ADC.ADC_16bit import ADC_16b
from DAC import DAC
from DAC.DAC_4bit import DAC_4b 
from DAC.DAC_1bit import DAC_1b 

from MRR_DPE import MRR_DPE
from PD import PD
from ReductionNetwork import RN
from SOA import SOA
from Shifter import Shifter
from VCSEL import VCSEL
from VoltageAdder import VoltageAdder
import pandas as pd
import sys
sys.path.append(".")
from Config import *


random_seed = 1
torch.manual_seed(random_seed)

def LIGHTBULB_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type):
    # cacha latency parameters
    cacheMissRatioDf = pd.read_csv(CACHE_MISS_RATIO_LUT_PATH)
    cacheParameters = pd.read_csv(CACHE_PARAMETER_LUT_PATH)
    l1_latency = cacheParameters[cacheParameters['cache']=='l1']
    l2_latency = cacheParameters[cacheParameters['cache']=='l2']

    # MRR DPE Latencies
    dac_latency = 0
    prop_latency = 0
    input_actuation_latency = 0
    weight_actuation_latency = 0
    dpe_latency = 0 # sum of prop, input_actuation and weight_actuation latency
    vcsel_latency = 0
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
    pd_energy = 0
    dac_energy = 0
    adc_energy = 0
    vcsel_energy = 0

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


    # C = 4
    # D = 4
    # K = 4
    # N = 4 # size of DPE
    # M = 4 # Number of DPEs in a DPU
    # Y = 10 # Number of DPUs
    I = torch.randn(C,K)
    W = torch.randn(K,D)
    O = torch.zeros(C,D)
    sup_act_precision = 1
    sup_wt_precision = 1
    
    B = 4  # Here B is the number of DPEs to support different bit shifted numbers
    #! Intra DPU Sharing
    # print("Input ", I)
    # print("Weight ", W)
    num_of_act_bit_slice = math.ceil(act_precision/sup_act_precision)
    num_of_wt_bit_slice = math.ceil(wt_precision/sup_wt_precision)
    data_rate = 5
    miss_ratio = cacheMissRatioDf.loc[(cacheMissRatioDf['C']==C) & (cacheMissRatioDf['D']==D) & (cacheMissRatioDf['K']==K) & (cacheMissRatioDf['dataflow']== 'OS')]
    # components 
    dpe_obj = MRR_DPE(N,data_rate)
    rn_obj = RN(reduction_network_type)
    wgt_dac_obj = DAC_1b()
    act_dac_obj = DAC_1b()
    adc_obj = ADC_8bit.ADC_8b()
    pd_obj = PD()
    shifter_obj = Shifter()

    ps_to_sec = 1e-12
    ns_to_sec = 1e-9
    us_to_sec = 1e-6

    fJ_to_J = 1e-15
    pJ_to_J = 1e-12
    nJ_to_J = 1e-9
    mW_to_W = 1e-3

    # #! Intra DPU Sharing 
    # print("Input ", I)
    # print("Weight ", W)
    sup_act_precision = 4
    B = 4  # Here B is the number of DPEs to support different bit shifted numbers
  

    for bit_slice in range(num_of_act_bit_slice*num_of_wt_bit_slice): 
        O = torch.zeros(C,D)
        for c in range(0, C, Y):
            for d in range(0, D, M):
                temp_partial_sum_counter = 0
                for k in range(0, K, N):
                    i_slice = I[c: min(c+Y,C), k : min(k + N, K)]
                    w_slice = W[k : min(k + N, K), d:min(d+M,D)]
                    w_slice = w_slice.T

                    # latency parameters calculations
                    dac_latency +=  act_dac_obj.latency*ns_to_sec
                    # weight_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                    input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec # ! Weight actuation latency is same as input actuation latency
                    prop_latency +=  dpe_obj.get_prop_latency()
                    pd_latency += pd_obj.latency*ps_to_sec
                    
                    for dpu_idx in range(min(c+Y,C)-c):
                        dpu_i_slice = i_slice[dpu_idx,:]
                        dac_energy += act_dac_obj.energy*pJ_to_J*torch.numel(dpu_i_slice)
                        dac_energy += wgt_dac_obj.energy*pJ_to_J*torch.numel(w_slice)
                        
                        weight_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec*torch.numel(w_slice) # J # ! OXBNN uses electro optic tuning for weight and input actuation
                        
                        pd_energy += pd_obj.energy*fJ_to_J*w_slice.shape[0]
                        
                        dpu_i_slice = dpu_i_slice.T.repeat(min(d+M,D)-d,1)
                        input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec*torch.numel(dpu_i_slice) # J
                        dpu_w_slice = w_slice
                        psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)    
                        O[c+dpu_idx,d:min(d+M,D)] = psum_dpu+O[c+dpu_idx,d:min(d+M,D)]
                        
                        # ! No partial sums are generated due to PCA based reduction
                        psum_access_latency += 0
                        psum_access_energy += 0
                      
                adc_latency += adc_obj.latency*ns_to_sec
                adc_energy += adc_obj.energy*pJ_to_J*torch.numel(psum_dpu)
                psum_reduction_latency += 0 
                psum_reduction_latency += shifter_obj.latency*ps_to_sec*torch.numel(psum_dpu)
                partial_sum_reduction_energy += shifter_obj.energy*fJ_to_J*torch.numel(psum_dpu)
                

    total_latency = dac_latency + input_actuation_latency + weight_actuation_latency + prop_latency + vcsel_energy + pd_latency + adc_latency + psum_access_latency + psum_reduction_latency
    total_energy = dac_energy + input_actuation_energy + weight_actuation_energy + vcsel_energy + pd_energy + adc_energy + psum_access_energy + partial_sum_reduction_energy

    latency_dict = {'reduction_network':reduction_network_type,'dataflow':'OS','propagation_latency':prop_latency, 'input_actuation_latency':input_actuation_latency, 'weight_actuation_latency':weight_actuation_latency,'dac_latency': dac_latency, 'pd_latency': pd_latency ,'adc_latency':adc_latency,'psum_access_latency':psum_access_latency, 'input_access_latency':input_access_latency, 'weight_access_latency':weight_access_latency, 'output_access_latency':output_access_latency, 'psum_reduction_latency':psum_reduction_latency, 'total_latency':total_latency}

    energy_dict = {'reduction_network':reduction_network_type,'dataflow':'OS','psum_access_energy': psum_access_energy,'input_actuation_energy':input_actuation_energy,'weight_actuation_energy':weight_actuation_energy, 'dac_energy':dac_energy, 'adc_energy':adc_energy, 'pd_energy': pd_energy ,'psum_reduction_energy': partial_sum_reduction_energy, 'dac_energy':dac_energy, 'adc_energy':adc_energy, 'total_energy': total_energy}
  
    return latency_dict, energy_dict