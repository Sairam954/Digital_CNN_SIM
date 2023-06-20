from ADC import ADC
from Config import *
from DAC import DAC
from MRR_DPE import *
from ReductionNetwork import RN
from VoltageAdder import VoltageAdder
import pandas as pd

accelerator_list = [AMM_IS_S_TREE,AMM_WS_S_TREE, AMM_OS_S_TREE, AMM_IS_ST_Tree_Ac, AMM_WS_ST_Tree_Ac, AMM_OS_ST_Tree_Ac, AMM_IS_STIFT,AMM_WS_STIFT, AMM_OS_STIFT,AMM_IS_PCA,AMM_WS_PCA, AMM_OS_PCA, MAM_IS_S_TREE,MAM_WS_S_TREE, MAM_OS_S_TREE, MAM_IS_ST_Tree_Ac, MAM_WS_ST_Tree_Ac, MAM_OS_ST_Tree_Ac, MAM_IS_STIFT,MAM_WS_STIFT, MAM_OS_STIFT,MAM_IS_PCA,MAM_WS_PCA, MAM_OS_PCA,AMM_RIS_S_TREE,AMM_RWS_S_TREE, AMM_ROS_S_TREE, AMM_RIS_ST_Tree_Ac, AMM_RWS_ST_Tree_Ac, AMM_ROS_ST_Tree_Ac, AMM_RIS_STIFT,AMM_RWS_STIFT, AMM_ROS_STIFT,AMM_RIS_PCA,AMM_RWS_PCA, AMM_ROS_PCA,MAM_RIS_S_TREE,MAM_RWS_S_TREE, MAM_ROS_S_TREE, MAM_RIS_ST_Tree_Ac, MAM_RWS_ST_Tree_Ac, MAM_ROS_ST_Tree_Ac, MAM_RIS_STIFT,MAM_RWS_STIFT, MAM_ROS_STIFT,MAM_RIS_PCA,MAM_RWS_PCA, MAM_ROS_PCA]

cacheParameters = pd.read_csv('C:\\Users\\SSR226\\Desktop\\DataflowTesting\\CacheUtils\\Cache_Parameters.csv')
l1_latency = cacheParameters[cacheParameters['cache']=='l1']
l2_latency = cacheParameters[cacheParameters['cache']=='l2']
l1_cache_area = l1_latency['area(mm2)'].values[0]
l2_cache_area = l2_latency['area(mm2)'].values[0]

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
    
    mrr_obj = MRR_DPE(dpe_size, data_rate)
    adc_obj = ADC()
    dac_obj = DAC()
    rn_obj = RN(reduction_network_type)
    va_obj = VoltageAdder()
    
    if vdp_type == 'AMM':
        no_of_mrrs = dpu_count*dpe_count*dpe_size*2
    elif vdp_type == 'MAM':
        no_of_mrrs = dpu_count*(dpe_count*dpe_size+dpe_count)
    
    mrr_area = no_of_mrrs*mrr_obj.area
    adc_area = dpu_count*dpe_count*adc_obj.area
    dac_area = no_of_mrrs*dac_obj.area
    if reduction_network_type == 'PCA':
        rn_area = va_obj.area*dpu_count
    else:
        rn_area = dpu_count*rn_obj.area
    cache_area = l1_cache_area*dpu_count+l2_cache_area
    
    total_area = mrr_area+adc_area+dac_area+rn_area+cache_area
    
    area_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'mrr_area':mrr_area,'adc_area':adc_area,'dac_area':dac_area,'rn_area':rn_area,'cache_area':cache_area,'total_area':total_area}
    tpc_area_result.append(area_dict)

area_df = pd.DataFrame(tpc_area_result)
area_df.to_csv('Area_Results.csv',index=False)
