from ADC.ADC_16bit import ADC_16b
from ADC.ADC_8bit import ADC_8b
from Config import *
import math
from DAC.DAC_1bit import DAC_1b
from DAC.DAC_4bit import DAC_4b

from MRR_DPE import MRR_DPE
from PD import PD
from ReductionNetwork import RN
from SOA import SOA
from VCSEL import VCSEL

accelerator_list = [TEST_HQNNA,TEST_HSCONNA,TEST_ROBIN_PO]

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
    mW_to_W = 1e-3
    laser_power_per_wavelength = 1.274274986e-3 # W
    if vdp_type=='HQNNA':
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_4b()
        adc_obj = ADC_16b()
        soa_obj = SOA()
        pd_obj = PD()
        # conv unit static power (MAM Architecture)
        weight_bank_mrrs = conv_dpe_count*conv_dpe_size
        input_bank_mrrs = conv_dpe_size
        no_of_mrrs = input_bank_mrrs+weight_bank_mrrs
        no_of_dacs = no_of_mrrs
        no_of_soas = conv_dpe_count
        no_of_pds = conv_dpe_count
        no_of_adc = math.ceil(conv_dpe_count/4)  
        laser_power = laser_power_per_wavelength*conv_dpe_size
        no_of_rn = 1
        conv_unit_power = no_of_rn*rn_obj.power*mW_to_W+ dpe_obj.weight_actuation_power*no_of_mrrs + dac_obj.power*no_of_dacs*mW_to_W + adc_obj.power*no_of_adc*mW_to_W + soa_obj.power*no_of_soas*mW_to_W + pd_obj.power*no_of_pds*mW_to_W + laser_power
        total_conv_unit_power = conv_unit_power*conv_dpu_count 
        print('HQNNA Conv Unit', conv_unit_power)
        # fc unit static power
        weight_bank_mrrs = fc_dpe_count*fc_dpe_size
        input_bank_mrrs = fc_dpe_size
        no_of_mrrs = input_bank_mrrs+weight_bank_mrrs
        no_of_dacs = no_of_mrrs
        no_of_soas = 0
        no_of_pds = fc_dpe_count
        no_of_adc = fc_dpe_count  
        laser_power = laser_power_per_wavelength*fc_dpe_size
        no_of_rn = 1
        fc_unit_power = no_of_rn*rn_obj.power*mW_to_W+dpe_obj.weight_actuation_power*no_of_mrrs + dac_obj.power*no_of_dacs*mW_to_W + adc_obj.power*no_of_adc*mW_to_W + soa_obj.power*no_of_soas*mW_to_W + pd_obj.power*no_of_pds*mW_to_W + laser_power
        total_fc_unit_power = fc_unit_power*fc_dpu_count 
        print('FC Unit', fc_unit_power)
        total_power = total_conv_unit_power + total_fc_unit_power
        print("HQNNA Power", total_power*1e3, "mW")
    elif vdp_type =="HSCONNA":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_1b()
        adc_obj = ADC_8b()
        soa_obj = SOA()
        pd_obj = PD()
        
        # sconna unit static power
        osm_bank_mrrs = dpe_count*dpe_size
        filter_bank_mrrs = dpe_count*dpe_size
        no_of_mrrs = filter_bank_mrrs+osm_bank_mrrs
        no_of_dacs = osm_bank_mrrs
        no_of_soas = 0
        no_of_pds = dpe_count
        no_of_adc = dpe_count  
        laser_power = laser_power_per_wavelength*dpe_size
        sconna_unit_power = dpe_obj.weight_actuation_power*osm_bank_mrrs + dac_obj.power*no_of_dacs*mW_to_W + adc_obj.power*no_of_adc*mW_to_W + soa_obj.power*no_of_soas*mW_to_W + pd_obj.power*no_of_pds*mW_to_W + laser_power
        print('SCONNA Unit Power', sconna_unit_power)
        total_sconna_unit_power = sconna_unit_power*dpu_count 
        total_power = total_sconna_unit_power
        print("HSCONNA Power", total_power*1e3, "mW")
    elif vdp_type =="ROBIN":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        act_dac_obj = DAC_1b()
        wgt_dac_obj = DAC_4b()
        adc_obj = ADC_16b()
        soa_obj = SOA()
        vcsel_obj = VCSEL()
        pd_obj = PD()
        
        # ROBIN unit static power
        weight_bank_mrrs = dpe_count*dpe_size
        input_bank_mrrs = dpe_count*dpe_size
        no_of_mrrs = weight_bank_mrrs+input_bank_mrrs
        no_of_vcsel = dpe_count
        no_of_pds = dpe_count
        no_of_adc = dpe_count  
        laser_power = laser_power_per_wavelength*dpe_size
        no_of_rn = 1
        robin_unit_power = no_of_rn*rn_obj.power*mW_to_W+dpe_obj.weight_actuation_power*no_of_mrrs + wgt_dac_obj.power*weight_bank_mrrs*mW_to_W + act_dac_obj.power*input_bank_mrrs*mW_to_W + adc_obj.power*no_of_adc*mW_to_W + vcsel_obj.power*no_of_vcsel*mW_to_W + pd_obj.power*no_of_pds*mW_to_W + laser_power
        print('ROBIN Unit Power', robin_unit_power)
        total_robin_unit_power = robin_unit_power*dpu_count 
        total_power = total_robin_unit_power
        print("ROBIN Power", total_power*1e3, "mW")