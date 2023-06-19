from os.path import isfile, join
from os import listdir, name
import pandas as pd
import torch 
import math
import matplotlib.pyplot as plt
from ADC import ADC
from Config import *
import torch.nn.functional as F

# Components import
from MRR_DPE import *
from ReductionNetwork import *
from DAC import *
from VoltageAdder import VoltageAdder


# accelerator_list = [AMM_IS_S_TREE,AMM_WS_S_TREE, AMM_OS_S_TREE, AMM_IS_ST_Tree_Ac, AMM_WS_ST_Tree_Ac, AMM_OS_ST_Tree_Ac, AMM_IS_STIFT,AMM_WS_STIFT, AMM_OS_STIFT,AMM_IS_PCA,AMM_WS_PCA, AMM_OS_PCA, MAM_IS_S_TREE,MAM_WS_S_TREE, MAM_OS_S_TREE, MAM_IS_ST_Tree_Ac, MAM_WS_ST_Tree_Ac, MAM_OS_ST_Tree_Ac, MAM_IS_STIFT,MAM_WS_STIFT, MAM_OS_STIFT,MAM_IS_PCA,MAM_WS_PCA, MAM_OS_PCA,AMM_RIS_S_TREE,AMM_RWS_S_TREE, AMM_ROS_S_TREE, AMM_RIS_ST_Tree_Ac, AMM_RWS_ST_Tree_Ac, AMM_ROS_ST_Tree_Ac, AMM_RIS_STIFT,AMM_RWS_STIFT, AMM_ROS_STIFT,AMM_RIS_PCA,AMM_RWS_PCA, AMM_ROS_PCA,MAM_RIS_S_TREE,MAM_RWS_S_TREE, MAM_ROS_S_TREE, MAM_RIS_ST_Tree_Ac, MAM_RWS_ST_Tree_Ac, MAM_ROS_ST_Tree_Ac, MAM_RIS_STIFT,MAM_RWS_STIFT, MAM_ROS_STIFT,MAM_RIS_PCA,MAM_RWS_PCA, MAM_ROS_PCA]

# TODO : Fix error in RIS and ROS output access counter calculation
 
accelerator_list = [AMM_RIS_S_TREE] 

model_precision = 8

print("Required Model Precision ", model_precision)
cnnModelDirectory = "CNNModels//"
modelList = [f for f in listdir(cnnModelDirectory) if isfile(join(cnnModelDirectory, f))]
modelList = ['GoogLeNet.csv']

ns_to_sec = 1e-9
us_to_sec = 1e-6

nJ_to_J = 1e-9
mW_to_W = 1e-3



# cacha latency parameters
cacheParameters = pd.read_csv('C:\\Users\\SSR226\\Desktop\\DataflowTesting\\CacheUtils\\Cache_Parameters.csv')
l1_latency = cacheParameters[cacheParameters['cache']=='l1']
l2_latency = cacheParameters[cacheParameters['cache']=='l2']
dram_latency = cacheParameters[cacheParameters['cache']=='dram']  
tpc_eval_result = []
tpc_latency_result = []
tpc_access_result = []
tpc_energy_result = []
for tpc in accelerator_list:
    architecture = tpc[0][NAME]
    batch_size = tpc[0][BATCH_SIZE]
    data_rate = tpc[0][BITRATE]
    dataflow = tpc[0][DATAFLOW]
    dpe_size = tpc[0][ELEMENT_SIZE]
    dpe_count = tpc[0][ELEMENT_COUNT]
    dpu_count = tpc[0][UNITS_COUNT]
    vdp_type = tpc[0][VDP_TYPE]
    acc_type = tpc[0][ACC_TYPE]
    reduction_network_type = tpc[0][REDUCTION_TYPE]

    # different latency parameters computed for GeMM execution
    # MRR DPE Latencies
    prop_latency = 0
    input_actuation_latency = 0
    weight_actuation_latency = 0
    dpe_latency = 0 # sum of prop, input_actuation and weight_actuation latency
    
    # cache access latencies 
    psum_access_latency = 0
    input_access_latency = 0
    weight_access_latency  = 0 
    output_access_latency = 0
    cache_access_latency = 0 # sum of psum, input, weight and output access latency
    
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
    dac_energy = 0
    adc_energy = 0
    
    # cache access energy
    weight_access_energy = 0
    input_access_energy = 0
    psum_access_energy = 0
    partial_sum_reduction_energy = 0
    output_access_energy = 0
    
    
    
    
    

    # storing metrics
    latency_dict = {}
    access_dict = {}
    energy_dict = {}
    
    for modelName in modelList:
        result = {}  
        print("Model being Processed ", modelName)
        nnModel = pd.read_csv(cnnModelDirectory+modelName)
        nnModel = nnModel.astype({"model_name": str, 'name': str, 'kernel_depth': int, 'kernel_height': int, 'kernel_width': int,	'tensor_count': int, 'input_shape': str,
                             'output_shape': str, 'tensor_shape': str,	'input_height': int,	'input_width': int, 'input_depth': int, 'output_height': int, 'output_width': int, 'output_depth': int})
        nnModel = pd.concat([nnModel]*batch_size, ignore_index=True)
        
        cacheMissRatioDf = pd.read_csv('C:\\Users\\SSR226\\Desktop\\MRRCNNSIM\\CacheUtils\\Miss_Ratio_Analysis1.csv')
        for idx in nnModel.index:
            layer_type = nnModel[LAYER_TYPE][idx]
            model_name = nnModel[MODEL_NAME][idx]
            kernel_depth = nnModel[KERNEL_DEPTH][idx]
            kernel_width = nnModel[KERNEL_WIDTH][idx]
            kernel_height = nnModel[KERNEL_HEIGHT][idx]
            tensor_count = nnModel[TENSOR_COUNT][idx]
            input_shape = nnModel[INPUT_SHAPE][idx]
            output_shape = nnModel[OUTPUT_SHAPE][idx]
            tensor_shape = nnModel[TENSOR_SHAPE][idx]
            input_height = nnModel[INPUT_HEIGHT][idx]
            input_width = nnModel[INPUT_WIDTH][idx]
            input_depth = nnModel[INPUT_DEPTH][idx]
            output_height = nnModel[OUTPUT_HEIGHT][idx]
            output_width = nnModel[OUTPUT_WIDTH][idx]
            output_depth = nnModel[OUTPUT_DEPTH][idx]
            stride_height = 1
            stride_width = 1
            # print('Layer', layer_type)
            in_channels = kernel_depth
            out_channels = kernel_height * kernel_width * in_channels
            out_height = (input_height - kernel_height) // stride_height + 1
            out_width = (input_width - kernel_width) // stride_width + 1
            inp = torch.randn(1, in_channels, input_height, input_width)
            w = torch.randn(tensor_count, kernel_depth, kernel_height, kernel_width)
            
            # Tranformation of convolutions and Fully connected layer operations into GEMM
            if layer_type=='Conv2D' or layer_type=='PointWiseConv': 
                toeplitz_input = torch.nn.functional.unfold(inp, kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))
                toeplitz_input = toeplitz_input.view(out_channels, out_height*out_width)
                toeplitz_w = w.view(w.size(0), -1)
            elif layer_type=='Dense':
                toeplitz_input = w.flatten().view(-1, 1)
                toeplitz_w = w.flatten().view(1, -1)
            toeplitz_w = torch.transpose(toeplitz_w, 0 , 1)
            # output = toeplitz_w @ toeplitz_input
            D = toeplitz_w.shape[1]
            C = toeplitz_input.shape[0]
            K = toeplitz_input.shape[1]
            # D = 100
            # C = 100
            # K = 100
            O = torch.zeros(C, D)
            A = toeplitz_w
            B = toeplitz_input
            X = dpe_size
            M = dpe_count
            Y = dpu_count
            # create the matrices 
            I = torch.randn(C,K)
            W = torch.randn(K,D)
            O = torch.zeros(C,D)
            
            print('I', I.shape)
            print('W', W.shape)
            print('O', O.shape)
            
            # miss ratio for the given dataflow and C, K, D combination 
            miss_ratio = cacheMissRatioDf.loc[(cacheMissRatioDf['C']==C) & (cacheMissRatioDf['D']==D) & (cacheMissRatioDf['K']==K) & (cacheMissRatioDf['dataflow']== dataflow)]
            print("Miss Ratio", miss_ratio)
            # obj of components needed for calculating latency and energy 
            dpe_obj = MRR_DPE(X,data_rate)
            rn_obj = RN()
            dac_obj = DAC()
            adc_obj = ADC()
            va_obj = VoltageAdder()
           
            # depending on dataflow perform the computations    
            if dataflow == 'WS':
                for d in range(0,D,Y):
                    for k in range(0, K, X):
                        w_slice = W[k : min(k + X, K), d:min(d+Y,D)]
                        weight_access_counter += torch.numel(w_slice)
                        weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                        weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec
                        
                        weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                        dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                        for c in range(0,C,M):
                            i_slice = I[c: min(c+M,C), k : min(k + X, K)]
                            input_access_counter += torch.numel(i_slice)
                            input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                            input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                            
                            input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                            input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                            prop_latency +=  dpe_obj.get_prop_latency()
                            for dpu_idx in range(min(d+Y,D)-d):
                                dpu_w_slice = w_slice[:,dpu_idx]
                                dpu_w_slice = dpu_w_slice.T.repeat(min(c+M,C)-c,1)
                                psum_dpu = torch.einsum('ij,ij->i', i_slice, dpu_w_slice)
                                # ! generated psum are sent to the cache and later brought back for accumulation, square mapping has to follow this
                                # ! Each partial sum belongs to different DPU, therefore it is write and read access 
                                O[c:c+M,d+dpu_idx] = psum_dpu+O[c:c+M,d+dpu_idx]
                                
                                if reduction_network_type == 'PCA':
                                    psum_access_counter += 0
                                    psum_access_latency += 0
                                    adc_energy += 0
                                    psum_access_energy += 0
                                else:
                                    psum_access_counter += 2*torch.numel(psum_dpu)
                                    psum_access_latency += 2*torch.numel(psum_dpu)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    
                                
                            
                            # Latency Calculation 
                            # ! Keep all latency in seconds 
                            # The loop with dpu_idx corresponds to dot product operation in all the DPU so it just the time taken to perform dot product operation 
                            # it will be just the propogation latency of light in the waveguide including the MRRs 
                            # since all DPUs and the DPEs inside them work in parallel the prop_latency of a DPE is equal to that of all DPUs  
                            
                            # ! FINISH THE PSUM PORTION AFTER OTHER latencies
                            # Each DPU generate psums, which will have two latency components 1)psum reduction latency 2)psum access latency 
                            # ! Mapping plays a role in psum accumulation based on 1) Inter DPU communication is allowed 2) Inter DPU Communication is not allowed 
                            # ! For the analysis there are two types of mappings which can achieve (2), the loops here are for square mapping
                            # ! In square mapping with electrical RNs, each DPU needs to send PSUM to memory and get it back for reduction
                            # ! 
                            # 1) Depends on the partial sum reduction network 
                            # S_Tree RN does not have capabikity of temporal accumulation, but the current mapping doesnot require spatial accumulation
                            # Therefore when RN is S_Tree, psum are sent to the cache and later brought back for reduction
                            
                            # i_slice has two latency components associated with it 1) accessing the input values from cache 2) Actuation Latency of MRR 
                            # 1) Depending on the dataflow and miss ratio latency will be determined, all DPUs work in parallel so the latency is overlapped 
                            
                        # w_slice has two latency components 1) accessing the input values from cache 2) Actuation Latency of MRR
                        # 1) Depending on the dataflow and miss ratio latency will be determined, all DPUs work in parallel so the latency is overlapped
                        
                    # ! each time a column is completed in WS, then C output pixels are computed and each pixel requires ceil(K/X) psum accumulations 
                    # ! we assume a redcution network for each dpu, therefore output pixel reduction is done in parallel
                    output_access_counter += C*(min(d+Y,D)-d)
                    psum_reduction_latency +=C*(min(d+Y,D)-d)*rn_obj.get_reduction_latency(math.ceil(K/X),1)
                    output_access_latency += C*(min(d+Y,D)-d)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                    
                    #! PCA case, for latency and energy conversion is taken care by the get reduction latency function
                    partial_sum_reduction_energy += C*(min(d+Y,D)-d)*rn_obj.get_reduction_latency(math.ceil(K/X),1)*rn_obj.power
                    output_access_energy +=  C*(min(d+Y,D)-d)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                    #! PCA case, psums are converted to digital values only when final result is obtained
                    if reduction_network_type=='PCA':
                        adc_energy += C*(min(d+Y,D)-d)*adc_obj.energy # J
                    
                    
            elif dataflow == 'OS':
                for c in range(0, C, M):
                    for d in range(0, D, Y):
                        for k in range(0, K, X):
                            i_slice = I[c: min(c+M,C), k : min(k + X, K)]
                            w_slice = W[k : min(k + X, K), d:min(d+Y,D)]
                            
                            
                            # Access Counter
                            input_access_counter += torch.numel(i_slice)
                            weight_access_counter += torch.numel(w_slice)
                            
                            # Latency Calculation
                            input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                            weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                            
                            weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec
                            input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                            prop_latency +=  dpe_obj.get_prop_latency()
                            
                            # Energy Calculation
                            weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                            dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                            
                            input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                            input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                            
                            for dpu_idx in range(min(d+Y,D)-d):
                                dpu_w_slice = w_slice[:,dpu_idx]
                                dpu_w_slice = dpu_w_slice.T.repeat(min(c+M,C)-c,1)
                                psum_dpu = torch.einsum('ij,ij->i', i_slice, dpu_w_slice)
                                O[c:c+M,d+dpu_idx] = psum_dpu+O[c:c+M,d+dpu_idx]
                                if reduction_network_type == 'PCA':
                                    psum_access_counter += 0
                                    psum_access_latency += 0
                                    adc_energy += 0
                                    psum_access_energy += 0
                                else:
                                    psum_access_counter += 2*torch.numel(psum_dpu)
                                    psum_access_latency += 2*torch.numel(psum_dpu)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                 
                                
                        
                        # ! whenever a k loop ends it is completes computation of a single output pixel
                        psum_reduction_latency += 1*(min(d+Y,D)-d)*(min(c+M,C)-c)*rn_obj.get_reduction_latency(math.ceil(K/X),1)
                        output_access_latency += 1*(min(d+Y,D)-d)*(min(c+M,C)-c)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                        output_access_counter += 1*(min(d+Y,D)-d)*(min(c+M,C)-c)
                        
                        partial_sum_reduction_energy += (min(d+Y,D)-d)*(min(c+M,C)-c)*rn_obj.get_reduction_latency(math.ceil(K/X),1)*rn_obj.power
                        output_access_energy +=  (min(d+Y,D)-d)*(min(c+M,C)-c)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        #! PCA case, psums are converted to digital values only when final result is obtained
                        if reduction_network_type=='PCA':
                            adc_energy += (min(d+Y,D)-d)*(min(c+M,C)-c)*adc_obj.energy # J
                        
            elif dataflow == 'IS':
                for c in range(0,C,M):
                    for k in range(0, K, X):
                        i_slice = I[c: min(c+M,C), k : min(k + X, K)]
                        input_access_counter += torch.numel(i_slice)
                        input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                        input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec 
                        
                        input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                        input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                          
                        for d in range(0, D, Y):
                            w_slice = W[k : min(k + X, K), d:min(d+Y,D)]
                            
                            weight_access_counter += torch.numel(w_slice)
                            weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                            weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec
                            
                            prop_latency +=  dpe_obj.get_prop_latency()
                            
                            weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                            dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                                    
                            for dpu_idx in range(min(d+Y,D)-d):
                                dpu_w_slice = w_slice[:,dpu_idx]
                                dpu_w_slice = dpu_w_slice.T.repeat(min(c+M,C)-c,1)
                                psum_dpu = torch.einsum('ij,ij->i', i_slice, dpu_w_slice)
                                O[c:c+M,d+dpu_idx] = psum_dpu+O[c:c+M,d+dpu_idx]
                                if reduction_network_type == 'PCA':
                                    psum_access_counter += 0
                                    psum_access_latency += 0
                                    adc_energy += 0
                                    psum_access_energy += 0
                                else:
                                    psum_access_counter += 2*torch.numel(psum_dpu)
                                    psum_access_latency += 2*torch.numel(psum_dpu)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += torch.numel(psum_dpu)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                 
                    
                    # ! each time a ROW is completed in IS, then D output pixels are computed and each pixel requires ceil(K/X) psum accumulations 
                    # ! we assume a reduction network for each dpu, therefore output pixel reduction is done in parallel
                    psum_reduction_latency += D*(min(c+M,C)-c)*rn_obj.get_reduction_latency(math.ceil(K/X),1)
                    output_access_latency += D*(min(c+M,C)-c)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                    output_access_counter += D*(min(c+M,C)-c)
                    
                    partial_sum_reduction_energy += D*(min(c+M,C)-c)*rn_obj.get_reduction_latency(math.ceil(K/X),1)*rn_obj.power
                    output_access_energy +=  D*(min(c+M,C)-c)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                    #! PCA case, psums are converted to digital values only when final result is obtained
                    if reduction_network_type=='PCA':
                        adc_energy += D*(min(c+M,C)-c)*adc_obj.energy # J
                    
            elif dataflow == 'ROS':
                # ! need this variable for PCA case, PCA also requires spatial accumulation but the number of psums required ADC conversion is reduced by factor of folds.
                temp_adc_energy = 0
                for c in range(0, C, Y):
                    for d in range(0, D):
                        for k in range(0, K, X*M):
                            i_slice = I[c: min(c+Y,C), k : min(k + X*M, K)]
                            
                            i_temp = i_slice.T
                            w_slice = W[k : min(k + X*M, K), d]
                            w_temp = w_slice.repeat(min(c+Y,C)-c,1).T
                            # print("Weight Temp", w_temp.shape)
                            
                            #access counter
                            input_access_counter += torch.numel(i_slice)
                            weight_access_counter += torch.numel(w_slice)
                            
                            # Latency Calculation
                            input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                            weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                            input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                            weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec  
                            prop_latency +=  dpe_obj.get_prop_latency()
                            
                            # Energy Calculation
                            input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                            input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                            
                            weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                            dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                            
                            for dpu_idx in range(min(c+Y,C)-c):
                                dpu_w_slice = w_temp[:,dpu_idx]
                                dpu_w_slice = F.pad(dpu_w_slice, (0, max(0, X*M-dpu_w_slice.shape[0])), mode='constant', value=0)
                                dpu_w_slice = dpu_w_slice.reshape(M,X)

                                dpu_i_slice = i_temp[:,dpu_idx]
                                dpu_i_slice = F.pad(dpu_i_slice, (0, max(0, X*M-dpu_i_slice.shape[0])), mode='constant', value=0)
                                dpu_i_slice = dpu_i_slice.reshape(M,X)


                                # PD or spatial reduction
                                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
                                # spatial reduction of psum using Tree based RN
                                reduced_psum_dpu = sum(psum_dpu)
                                O[c+dpu_idx,d] = reduced_psum_dpu+O[c+dpu_idx,d]
                                # ! In rectangular mapping, Spatial Reduction at DPEs of a DPU is possible, however, S_Tree cannot perform tempporal reduction
                                if reduction_network_type == 'S_Tree':
                                    psum_access_latency += 2*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    psum_access_counter += 2
                                    psum_access_energy += (l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += (l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                elif reduction_network_type == 'PCA':
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    temp_adc_energy += torch.numel(psum_dpu)*adc_obj.energy  
                                       
                                else:
                                # ! S_Tree_Acc and STIFT can perform temporal reduction at DPEs of a DPU so psum is not send and accessed from cache
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                
                                
                        # ! Spatial and Temoral reduction happens at DPEs of DPU
                        folds = math.ceil(K/X)
                        if torch.numel(psum_dpu)<folds:
                            folds=1
                            
                        
                        output_access_latency += (min(c+Y,C)-c)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                        output_access_counter += (min(c+Y,C)-c)
                        
                        output_access_energy += (min(c+Y,C)-c)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        
                        if reduction_network_type=='PCA':
                            adc_energy += temp_adc_energy/folds # J
                            psum_reduction_latency += va_obj.latency
                            partial_sum_reduction_energy += va_obj.latency*va_obj.power*torch.numel(psum_dpu)
                        else:
                            psum_reduction_latency +=rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)
                            partial_sum_reduction_energy += rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)*rn_obj.power
                            
            elif dataflow == 'RWS':
                temp_output_access_counter  = 0
                temp_adc_energy = 0
                for d in range(0,D):
                    for k in range(0, K, X*M):
                        w_slice = W[k : min(k + X*M, K), d]
                        weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                        weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec  
                        weight_access_counter += torch.numel(w_slice)  
                        
                        weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                        dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                        
                           
                        for c in range(0,C,Y):
                        
                            i_slice = I[c: min(c+Y,C), k : min(k + X*M, K)]
                            i_slice = i_slice.T # each column will belong to a DPU
                            
                            
                            input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                            input_access_counter += torch.numel(i_slice)
                            input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                            
                            input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                            input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                            
                            
                            temp = w_slice.repeat(min(c+Y,C)-c,1).T
                            prop_latency +=  dpe_obj.get_prop_latency()
                            for dpu_idx in range(min(c+Y,C)-c):

                                dpu_i_slice = i_slice[:,dpu_idx]
                                # padding the tensors with zero to use all the DPEs in DPU
                                dpu_i_slice = F.pad(dpu_i_slice, (0, max(0, X*M-dpu_i_slice.shape[0])), mode='constant', value=0)
                                dpu_i_slice = dpu_i_slice.reshape(M,X)


                                dpu_w_slice = temp[:,dpu_idx]
                                dpu_w_slice = F.pad(dpu_w_slice, (0, max(0, X*M-dpu_w_slice.shape[0])), mode='constant', value=0)
                                dpu_w_slice = dpu_w_slice.reshape(M,X)

                                # PD or spatial reduction
                                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)
                                # spatial reduction of psum using Tree based RN
                                reduced_psum_dpu = sum(psum_dpu)
                                O[c+dpu_idx,d] = reduced_psum_dpu+O[c+dpu_idx,d]
                                if reduction_network_type == 'S_Tree':
                                    psum_access_latency += 2*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    psum_access_counter += 2
                                    psum_access_energy += (l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += (l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                elif reduction_network_type == 'PCA':
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    temp_adc_energy += torch.numel(psum_dpu)*adc_obj.energy        
                                else:
                                # ! S_Tree_Acc and STIFT can perform temporal reduction at DPEs of a DPU so psum is not send and accessed from cache
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                            temp_output_access_counter += (min(c+Y,C)-c)
                    folds = math.ceil(K/(X*M))
                    if torch.numel(psum_dpu)<folds:
                            folds=1   
                    if reduction_network_type=='PCA':
                            adc_energy += temp_adc_energy/folds # J
                            psum_reduction_latency += va_obj.latency
                            partial_sum_reduction_energy += va_obj.latency*va_obj.power*torch.numel(psum_dpu)
                    else:
                        psum_reduction_latency +=rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)
                        partial_sum_reduction_energy += rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)*rn_obj.power
                            
                output_access_counter += (temp_output_access_counter/folds)
                output_access_latency += (temp_output_access_counter/folds)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                
                output_access_energy +=(temp_output_access_counter/folds)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                       
           
            elif dataflow == 'RIS':
                temp_output_access_counter = 0
                temp_adc_energy = 0
                for c in range(0,C):
                    for k in range(0, K, X*M):
                        i_slice = I[c, k : min(k + X*M, K)]
                        
                        input_access_latency += torch.numel(i_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec 
                        input_access_counter += torch.numel(i_slice)
                        input_actuation_latency += dpe_obj.input_actuation_latency*ns_to_sec
                        
                        input_access_energy += torch.numel(i_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                        dac_energy += torch.numel(i_slice)*dac_obj.energy # J
                        input_actuation_energy += dpe_obj.input_actuation_power*dpe_obj.input_actuation_latency*ns_to_sec # J
                            
                        
                        for d in range(0, D, Y):
                            w_slice = W[k : min(k + X*M, K), d:min(d+Y,D)]
                            
                            weight_access_latency += torch.numel(w_slice)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                            weight_actuation_latency += dpe_obj.thermo_optic_tuning_latency*us_to_sec  
                            weight_access_counter += torch.numel(w_slice)     
                            prop_latency +=  dpe_obj.get_prop_latency()
                            temp = i_slice.repeat(min(d+Y,D)-d,1).T
                            
                            weight_access_energy += torch.numel(w_slice)*(l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                            weight_actuation_energy += dpe_obj.weight_actuation_power*dpe_obj.thermo_optic_tuning_latency*us_to_sec # J
                            dac_energy += torch.numel(w_slice)*dac_obj.energy # J
                            
                            for dpu_idx in range(min(d+Y,D)-d):
                                dpu_w_slice = w_slice[:,dpu_idx]
                                # padding the tensors with zero to use all the DPEs in DPU
                                dpu_w_slice = F.pad(dpu_w_slice, (0, max(0, X*M-dpu_w_slice.shape[0])), mode='constant', value=0)
                                dpu_w_slice = dpu_w_slice.reshape(M,X)

                                dpu_i_slice = temp[:,dpu_idx]
                                dpu_i_slice = F.pad(dpu_i_slice, (0, max(0, X*M-dpu_i_slice.shape[0])), mode='constant', value=0)
                                dpu_i_slice = dpu_i_slice.reshape(M,X)
                                # Spatial reduction at PD
                                psum_dpu = torch.einsum('ij,ij->i', dpu_i_slice, dpu_w_slice)

                                # spatial reduction of psum using Tree based RN
                                reduced_psum_dpu = sum(psum_dpu)
                                O[c,d+dpu_idx] = reduced_psum_dpu+O[c,d+dpu_idx]
                                if reduction_network_type == 'S_Tree':
                                    psum_access_latency += 2*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
                                    psum_access_counter += 2
                                    psum_access_energy += (l1_latency['energy_read(nJ)'].values[0]+l2_latency['energy_read(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    psum_access_energy += (l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                                elif reduction_network_type == 'PCA':
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    temp_adc_energy += torch.numel(psum_dpu)*adc_obj.energy        
                                else:
                                # ! S_Tree_Acc and STIFT can perform temporal reduction at DPEs of a DPU so psum is not send and accessed from cache
                                    psum_access_latency += 0
                                    psum_access_counter += 0
                                    psum_access_energy += 0
                                    adc_energy += torch.numel(psum_dpu)*adc_obj.energy # J
                            temp_output_access_counter += (min(d+Y,D)-d)
                    folds = math.ceil(K/(X*M))
                    if torch.numel(psum_dpu)<folds:
                            folds=1
                    if reduction_network_type=='PCA':
                            adc_energy += temp_adc_energy/folds # J
                            psum_reduction_latency += va_obj.latency
                            partial_sum_reduction_energy += va_obj.latency*va_obj.power*torch.numel(psum_dpu)
                    else:
                        psum_reduction_latency +=rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)
                        partial_sum_reduction_energy += rn_obj.get_reduction_latency(torch.numel(psum_dpu),folds)*rn_obj.power 
                      
                output_access_counter += temp_output_access_counter/folds
                output_access_latency += (temp_output_access_counter/folds)*(l1_latency['ti(ns)'].values[0]+l2_latency['ti(ns)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*ns_to_sec
       
                output_access_energy +=(temp_output_access_counter/folds)*(l1_latency['energy_write(nJ)'].values[0]+l2_latency['energy_write(nJ)'].values[0]*miss_ratio['l1_miss_ratio'].values[0])*nJ_to_J 
                
        # print all the latency parameters
        # MRR DPE Latencies
        print('Propagation Latency',prop_latency)
        print('Input Actuation Latency',input_actuation_latency)    
        print('Weight Actuation Latency',weight_actuation_latency)
        
        # cache access latencies 
        print("Psum Access Latency",psum_access_latency)
        print("Input Access Latency",input_access_latency)
        print("Weight Access Latency",weight_access_latency)
        print("Output Access Latency",output_access_latency)
        
        # Psum reduction latency at RN
        print("Psum Reduction Latency",psum_reduction_latency)
        
        dpe_latency = prop_latency+input_actuation_latency+weight_actuation_latency
        cache_access_latency = psum_access_latency+input_access_latency+weight_access_latency+output_access_latency
        psum_reduction_latency = psum_reduction_latency
        total_access = psum_access_counter+input_access_counter+weight_access_counter+output_access_counter
        total_latency = dpe_latency+cache_access_latency+psum_reduction_latency
        total_energy = input_actuation_energy+weight_actuation_energy+input_access_energy+weight_access_energy+output_access_energy+psum_access_energy+dac_energy+adc_energy+partial_sum_reduction_energy
        
        print("Total Latency",total_latency)                  
        
            

        latency_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'propagation_latency':prop_latency, 'input_actuation_latency':input_actuation_latency, 'weight_actuation_latency':weight_actuation_latency, 'psum_access_latency':psum_access_latency, 'input_access_latency':input_access_latency, 'weight_access_latency':weight_access_latency, 'output_access_latency':output_access_latency, 'psum_reduction_latency':psum_reduction_latency, 'total_latency':total_latency}
        tpc_latency_result.append(latency_dict)
        
        access_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'psum_access_counter':psum_access_counter, 'input_access_counter':input_access_counter, 'weight_access_counter':weight_access_counter, 'output_access_counter':output_access_counter, 'total_access':total_access}
        tpc_access_result.append(access_dict)
        
        energy_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'psum_access_energy': psum_access_energy,'input_actuation_energy':input_actuation_energy,'weight_actuation_energy':weight_actuation_energy,'input_access_energy':input_access_energy,'weight_access_energy':weight_access_energy,'output_access_energy':output_access_energy, 'psum_reduction_energy': partial_sum_reduction_energy, 'dac_energy':dac_energy, 'adc_energy':adc_energy, 'total_energy': total_energy}
        tpc_energy_result.append(energy_dict)
latency_df = pd.DataFrame(tpc_latency_result)
access_df = pd.DataFrame(tpc_access_result)
energy_df = pd.DataFrame(tpc_energy_result)

latency_df.to_csv('tpc_latency_result.csv',index=False)
access_df.to_csv('tpc_access_result.csv',index=False)
energy_df.to_csv('tpc_energy_result.csv', index=False)

