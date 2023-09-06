from matplotlib.font_manager import weight_dict
from ADC import ADC
from ConfigHEANA import *
from DAC import DAC
from MRR_DPE import *
from ReductionNetwork import RN
from VoltageAdder import VoltageAdder
import pandas as pd

accelerator_list = [AMW_OS_PCA_LS, MAW_OS_PCA_LS, AMW5_OS_PCA_LS, MAW5_OS_PCA_LS, AMW10_OS_PCA_LS, MAW10_OS_PCA_LS, AMW_OS_S_Tree_LS, MAW_OS_S_Tree_LS, AMW5_OS_S_Tree_LS, MAW5_OS_S_Tree_LS, AMW10_OS_S_Tree_LS, MAW10_OS_S_Tree_LS, AMW_WS_PCA_LS, MAW_WS_PCA_LS, AMW5_WS_PCA_LS, MAW5_WS_PCA_LS, AMW10_WS_PCA_LS, MAW10_WS_PCA_LS, AMW_WS_S_Tree_LS, MAW_WS_S_Tree_LS, AMW5_WS_S_Tree_LS, MAW5_WS_S_Tree_LS, AMW10_WS_S_Tree_LS, MAW10_WS_S_Tree_LS, AMW_IS_PCA_LS, MAW_IS_PCA_LS, AMW5_IS_PCA_LS, MAW5_IS_PCA_LS, AMW10_IS_PCA_LS, MAW10_IS_PCA_LS, AMW_IS_S_Tree_LS, MAW_IS_S_Tree_LS, AMW5_IS_S_Tree_LS, MAW5_IS_S_Tree_LS, AMW10_IS_S_Tree_LS, MAW10_IS_S_Tree_LS, HEANA_OS_PCA_LS, HEANA5_OS_PCA_LS, HEANA10_OS_PCA_LS, HEANA_WS_PCA_LS, HEANA5_WS_PCA_LS, HEANA10_WS_PCA_LS, HEANA_IS_PCA_LS, HEANA5_IS_PCA_LS, HEANA10_IS_PCA_LS]
cacheParameters = pd.read_csv('CacheUtils\\Cache_Parameters.csv')
l1_latency = cacheParameters[cacheParameters['cache']=='l1']
l2_latency = cacheParameters[cacheParameters['cache']=='l2']
l1_cache_power = l1_latency['leakage_power(mW)'].values[0]
l2_cache_power = l2_latency['leakage_power(mW)'].values[0]

tpc_area_result = []

for tpc in accelerator_list:
    area_dict = {}
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
    print("Architecture ", architecture)
    mrr_obj = MRR_DPE(dpe_size, data_rate)
    if vdp_type == 'HEANA' and dataflow == 'OS':
        data_rate = 1
    adc_obj = ADC(data_rate)
    dac_obj = DAC()
    rn_obj = RN(reduction_network_type)
    va_obj = VoltageAdder()
    
    if vdp_type == 'AMM':
        # no_of_mrrs = dpu_count*dpe_count*dpe_size*2
        input_actuation_mrr = dpu_count*dpe_count*dpe_size
        weight_actuation_mrr = dpu_count*dpe_count*dpe_size
        no_of_mrrs = input_actuation_mrr+weight_actuation_mrr
        mrr_power = (input_actuation_mrr*mrr_obj.input_actuation_power)+(weight_actuation_mrr*mrr_obj.weight_actuation_power)
    elif vdp_type == 'MAM':
        
        weight_actuation_mrr = dpu_count*dpe_count*dpe_size
        input_actuation_mrr = dpu_count*dpe_count
        no_of_mrrs = input_actuation_mrr+weight_actuation_mrr
        mrr_power = (input_actuation_mrr*mrr_obj.input_actuation_power)+(weight_actuation_mrr*mrr_obj.weight_actuation_power)
    elif vdp_type == 'HEANA':
        no_of_mrrs = dpu_count*dpe_count*dpe_size
        mrr_power = no_of_mrrs*mrr_obj.input_actuation_power
        

    adc_power = dpu_count*dpe_count*adc_obj.power
    dac_power = no_of_mrrs*dac_obj.power
    if reduction_network_type == 'PCA':
        rn_power = va_obj.power*dpu_count
    else:
        rn_power = dpu_count*rn_obj.power
    cache_power = l1_cache_power*dpu_count+l2_cache_power
    
    total_power = mrr_power+adc_power+dac_power+rn_power+cache_power
    print("Total Power ", total_power)
    area_dict = {'DPU':architecture,'reduction_network':reduction_network_type,'dataflow':dataflow,'mrr_power':mrr_power,'adc_power':adc_power,'dac_power':dac_power,'rn_area':rn_power,'cache_power':cache_power,'total_power':total_power}
    tpc_area_result.append(area_dict)

area_df = pd.DataFrame(tpc_area_result)
area_df.to_csv('Static_Power_Results.csv',index=False)
