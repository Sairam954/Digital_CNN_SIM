from ADC import ADC
from ConfigHEANA import *
from DAC import DAC
from MRR_DPE import *
from ReductionNetwork import RN
from VoltageAdder import VoltageAdder
import pandas as pd

accelerator_list = [AMW_WS_S_TREE_LS, MAW_WS_S_TREE_LS, HEANA_WS_PCA_LS]

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
    print("Architecture ", architecture)
    mrr_obj = MRR_DPE(dpe_size, data_rate)
    adc_obj = ADC(data_rate)
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
    print("Total Area ", total_area)
    area_dict = {'DPU':vdp_type,'reduction_network':reduction_network_type,'dataflow':dataflow,'mrr_area':mrr_area,'adc_area':adc_area,'dac_area':dac_area,'rn_area':rn_area,'cache_area':cache_area,'total_area':total_area}
    tpc_area_result.append(area_dict)

area_df = pd.DataFrame(tpc_area_result)
area_df.to_csv('Area_Results.csv',index=False)
