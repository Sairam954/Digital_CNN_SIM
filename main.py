from os.path import isfile, join
from os import listdir, name
import pandas as pd
import torch 
import math
import matplotlib.pyplot as plt
from ADC.ADC import ADC
from Config import *
import torch.nn.functional as F
import datetime


# Components import
from MRR_DPE import *
from Mapper.HQNNA_Conv import HQNNA_Conv_run
from Mapper.HQNNA_FC import HQNNA_FC_run
from Mapper.HSCONNA import HSCONNA_run
from Mapper.OXBNN import OXBNN_run
from Mapper.ROBIN import ROBIN_run
from ReductionNetwork import *
from DAC import *
from VoltageAdder import VoltageAdder


accelerator_list = [TEST_HSCONNA, TEST_SCONNA, TEST_HQNNA, TEST_ROBIN_PO, TEST_ROBIN_EO, TEST_OXBNN]

cnnModelDirectory = "CNNModels//Sample//"
modelList = [f for f in listdir(cnnModelDirectory) if isfile(join(cnnModelDirectory, f))]
modelList = ['GoogLeNet.csv']

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
    conv_dpe_size = tpc[0][CONV_ELEMENT_SIZE]
    conv_dpe_count = tpc[0][CONV_ELEMENT_COUNT]
    conv_dpu_count = tpc[0][CONV_UNITS_COUNT]
    fc_dpe_size = tpc[0][FC_ELEMENT_SIZE]
    fc_dpe_count = tpc[0][FC_ELEMENT_COUNT]
    fc_dpu_count = tpc[0][FC_UNITS_COUNT]
    vdp_type = tpc[0][VDP_TYPE]
    reduction_network_type = tpc[0][REDUCTION_TYPE]
    print("Architecture ", architecture, "Dataflow ", dataflow, "Reduction Network", reduction_network_type)
    
    #! Assertions to Ensure that the input configuration is valid
    if vdp_type == "HQNNA":
        assert (dpe_count%4 == 0), "Each Cluster should have equal number of DPEs"
        
    # MRR DPE Latencies
    dac_latency = 0
    prop_latency = 0
    input_actuation_latency = 0
    weight_actuation_latency = 0
    dpe_latency = 0 # sum of prop, input_actuation and weight_actuation latency
    soa_latency = 0
    adc_latency = 0
    pd_latency = 0
    vcsel_latency = 0
    b_to_s_latency = 0

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
    vcsel_energy = 0
    b_to_s_energy = 0

    # cache access energy
    weight_access_energy = 0
    input_access_energy = 0
    psum_access_energy = 0
    psum_reduction_energy = 0
    output_access_energy = 0

    # Mrr Utilization
    used_mrr_counter = 0
    unused_mrr_counter = 0

    # Reduction Folds counter: To know how many times temporal reduction is used by a DPU        
    folds_counter = 0

    # storing metrics
    latency_dict = {}
    energy_dict = {}        


    for modelName in modelList:
        result = {}  
        print("Model being Processed ", modelName)
        nnModel = pd.read_csv(cnnModelDirectory+modelName)
        nnModel = nnModel.astype({"model_name": str, 'name': str, 'kernel_depth': int, 'kernel_height': int, 'kernel_width': int,	'tensor_count': int, 'input_shape': str,
                             'output_shape': str, 'tensor_shape': str,	'input_height': int,	'input_width': int, 'input_depth': int, 'output_height': int, 'output_width': int, 'output_depth': int})
        nnModel = pd.concat([nnModel]*batch_size, ignore_index=True)
        
        cacheMissRatioDf = pd.read_csv(CACHE_MISS_RATIO_LUT_PATH)
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
            act_precision = nnModel[ACT_PRECISION][idx]
            wt_precision = nnModel[WT_PRECISION][idx]
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
            N = dpe_size
            M = dpe_count
            Y = dpu_count
            
            if vdp_type == 'HQNNA':
                if layer_type == 'Conv2D' or layer_type=='PointWiseConv':
                    N = conv_dpe_size
                    M = conv_dpe_count
                    Y = conv_dpu_count
                    latency_dict, energy_dict = HQNNA_Conv_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                
                elif layer_type == 'Dense':
                    N = fc_dpe_size
                    M = fc_dpe_count
                    Y = fc_dpu_count
                    latency_dict, energy_dict = HQNNA_FC_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                # update global latency and energy
                prop_latency += latency_dict['propagation_latency']
                input_actuation_latency += latency_dict['input_actuation_latency']
                weight_actuation_latency += latency_dict['weight_actuation_latency']
                dac_latency += latency_dict['dac_latency']
                adc_latency += latency_dict['adc_latency']
                pd_latency += latency_dict['pd_latency']
                soa_latency += latency_dict['soa_latency']
                psum_access_latency += latency_dict['psum_access_latency']
                psum_reduction_latency += latency_dict['psum_reduction_latency']
                input_access_latency += 0
                weight_access_latency += 0
                output_access_latency += 0
                
                psum_access_energy += energy_dict['psum_access_energy']
                input_actuation_energy += energy_dict['input_actuation_energy']
                weight_actuation_energy += energy_dict['weight_actuation_energy']
                dac_energy += energy_dict['dac_energy']
                adc_energy += energy_dict['adc_energy']
                soa_energy += energy_dict['soa_energy']
                psum_reduction_energy += energy_dict['psum_reduction_energy']
                psum_access_energy += energy_dict['psum_access_energy']
                input_access_energy += 0
                weight_access_energy += 0
                output_access_energy += 0
                
            elif vdp_type == 'ROBIN':
                latency_dict, energy_dict = ROBIN_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                # update global latency and energy
                prop_latency += latency_dict['propagation_latency']
                input_actuation_latency += latency_dict['input_actuation_latency']
                weight_actuation_latency += latency_dict['weight_actuation_latency']
                dac_latency += latency_dict['dac_latency']
                adc_latency += latency_dict['adc_latency']
                pd_latency += latency_dict['pd_latency']
                vcsel_latency += latency_dict['vcsel_latency']
                psum_access_latency += latency_dict['psum_access_latency']
                psum_reduction_latency += latency_dict['psum_reduction_latency']
                input_access_latency += 0
                weight_access_latency += 0
                output_access_latency += 0
                
                psum_access_energy += energy_dict['psum_access_energy']
                input_actuation_energy += energy_dict['input_actuation_energy']
                weight_actuation_energy += energy_dict['weight_actuation_energy']
                dac_energy += energy_dict['dac_energy']
                adc_energy += energy_dict['adc_energy']
                vcsel_energy += energy_dict['vcsel_energy']
                psum_reduction_energy += energy_dict['psum_reduction_energy']
                psum_access_energy += energy_dict['psum_access_energy']
                input_access_energy += 0
                weight_access_energy += 0
                output_access_energy += 0
                
                
            elif vdp_type == 'LIGHTBULB':
                pass
            elif vdp_type == 'OXBNN':
                latency_dict, energy_dict = OXBNN_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                prop_latency += latency_dict['propagation_latency']
                input_actuation_latency += latency_dict['input_actuation_latency']
                weight_actuation_latency += latency_dict['weight_actuation_latency']
                dac_latency += latency_dict['dac_latency']
                adc_latency += latency_dict['adc_latency']
                pd_latency += latency_dict['pd_latency']
                psum_access_latency += latency_dict['psum_access_latency']
                psum_reduction_latency += latency_dict['psum_reduction_latency']
                input_access_latency += 0
                weight_access_latency += 0
                output_access_latency += 0
                
                psum_access_energy += energy_dict['psum_access_energy']
                input_actuation_energy += energy_dict['input_actuation_energy']
                weight_actuation_energy += energy_dict['weight_actuation_energy']
                dac_energy += energy_dict['dac_energy']
                adc_energy += energy_dict['adc_energy']
                psum_reduction_energy += energy_dict['psum_reduction_energy']
                psum_access_energy += energy_dict['psum_access_energy']
                input_access_energy += 0
                weight_access_energy += 0
                output_access_energy += 0
            elif vdp_type == 'SCONNA':
                latency_dict, energy_dict = HSCONNA_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                prop_latency += latency_dict['propagation_latency']
                input_actuation_latency += latency_dict['input_actuation_latency']
                weight_actuation_latency += latency_dict['weight_actuation_latency']
                b_to_s_latency += latency_dict['b_to_s_latency']
                dac_latency += latency_dict['dac_latency']
                adc_latency += latency_dict['adc_latency']
                pd_latency += latency_dict['pd_latency']
                psum_access_latency += latency_dict['psum_access_latency']
                psum_reduction_latency += latency_dict['psum_reduction_latency']
                input_access_latency += 0
                weight_access_latency += 0
                output_access_latency += 0
                
                psum_access_energy += energy_dict['psum_access_energy']
                input_actuation_energy += energy_dict['input_actuation_energy']
                weight_actuation_energy += energy_dict['weight_actuation_energy']
                b_to_s_energy += energy_dict['b_to_s_energy']
                dac_energy += energy_dict['dac_energy']
                adc_energy += energy_dict['adc_energy']
                psum_reduction_energy += energy_dict['psum_reduction_energy']
                psum_access_energy += energy_dict['psum_access_energy']
                input_access_energy += 0
                weight_access_energy += 0
                output_access_energy += 0
                
            elif vdp_type == 'HSCONNA':
                latency_dict, energy_dict = HSCONNA_run(C, D, K, N, M, Y, act_precision, wt_precision, reduction_network_type)
                prop_latency += latency_dict['propagation_latency']
                input_actuation_latency += latency_dict['input_actuation_latency']
                weight_actuation_latency += latency_dict['weight_actuation_latency']
                b_to_s_latency += latency_dict['b_to_s_latency']
                dac_latency += latency_dict['dac_latency']
                adc_latency += latency_dict['adc_latency']
                pd_latency += latency_dict['pd_latency']
                psum_access_latency += latency_dict['psum_access_latency']
                psum_reduction_latency += latency_dict['psum_reduction_latency']
                input_access_latency += 0
                weight_access_latency += 0
                output_access_latency += 0
                
                psum_access_energy += energy_dict['psum_access_energy']
                input_actuation_energy += energy_dict['input_actuation_energy']
                weight_actuation_energy += energy_dict['weight_actuation_energy']
                b_to_s_energy += energy_dict['b_to_s_energy']
                dac_energy += energy_dict['dac_energy']
                adc_energy += energy_dict['adc_energy']
                psum_reduction_energy += energy_dict['psum_reduction_energy']
                psum_access_energy += energy_dict['psum_access_energy']
                input_access_energy += 0
                weight_access_energy += 0
                output_access_energy += 0
                
                   
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
        
        
        # total_latency = dac_latency + input_actuation_latency + weight_actuation_latency + prop_latency + soa_latency + b_to_s_latency + vcsel_latency+ pd_latency + adc_latency + psum_access_latency + psum_reduction_latency
        total_latency = prop_latency+ soa_latency + b_to_s_latency + vcsel_latency+ pd_latency + adc_latency + psum_reduction_latency
        total_energy = dac_energy + input_actuation_energy + weight_actuation_energy + soa_energy + b_to_s_energy + vcsel_energy + pd_energy + adc_energy + psum_access_energy + psum_reduction_energy

        print("Total Latency",total_latency)   
        print("Total Energy",total_energy)

        latency_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'propagation_latency':prop_latency, 'input_actuation_latency':input_actuation_latency,
                        'weight_actuation_latency':weight_actuation_latency, 'psum_access_latency':psum_access_latency,'input_access_latency':input_access_latency, 
                        'weight_access_latency':weight_access_latency, 'output_access_latency':output_access_latency, 'psum_reduction_latency':psum_reduction_latency,'soa_latency':soa_latency, 
                        'b_to_s_latency': b_to_s_latency,'vcsel_latency':vcsel_latency,'total_latency':total_latency}
        tpc_latency_result.append(latency_dict)
    
        energy_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'psum_access_energy': psum_access_energy,'input_actuation_energy':input_actuation_energy,
                       'weight_actuation_energy':weight_actuation_energy,'input_access_energy':input_access_energy,'weight_access_energy':weight_access_energy,
                       'output_access_energy':output_access_energy, 'psum_reduction_energy': psum_reduction_energy, 'dac_energy':dac_energy, 'adc_energy':adc_energy, 
                       'soa_energy':soa_energy, 'b_to_s_energy': b_to_s_energy,'vcsel_energy':vcsel_energy,
                       'total_energy': total_energy}
        tpc_energy_result.append(energy_dict)
        latency_df = pd.DataFrame(tpc_latency_result)
        energy_df = pd.DataFrame(tpc_energy_result)


        # Get the current date and time
        # current_datetime = datetime.datetime.now()

        # # Convert the date and time to a string format
        # datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        datetime_string = "Sample"


        # add time log to the output file
        latency_df.to_csv('tpc_latency_result'+datetime_string+'.csv',index=False)
        energy_df.to_csv('tpc_energy_result'+datetime_string+'.csv', index=False)

