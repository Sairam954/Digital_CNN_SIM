from ADC.ADC_16bit import ADC_16b
from ADC.ADC_8bit import ADC_8b
from BtoSConverter import BToSConverter
from BtoSLookupTable import BtoSLookUpTable
from Config import *
import math
from DAC.DAC_1bit import DAC_1b
from DAC.DAC_4bit import DAC_4b

from MRR_DPE import MRR_DPE
from PD import PD
from ReductionNetwork import RN
from SOA import SOA
from VCSEL import VCSEL

accelerator_list = [TEST_HQNNA, TEST_HSCONNA, TEST_SCONNA, TEST_ROBIN_PO, TEST_ROBIN_EO, TEST_OXBNN]

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
    if vdp_type=='HQNNA':
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_4b()
        adc_obj = ADC_16b()
        soa_obj = SOA()
        pd_obj = PD()
        # conv unit area (MAM Architecture)
        weight_bank_mrrs = conv_dpe_count*conv_dpe_size
        input_bank_mrrs = conv_dpe_size
        no_of_mrrs = input_bank_mrrs+weight_bank_mrrs
        no_of_dacs = no_of_mrrs
        no_of_soas = conv_dpe_count
        no_of_pds = conv_dpe_count
        no_of_adc = math.ceil(conv_dpe_count/4)  
        no_of_rn = 1
        conv_unit_area = no_of_rn*rn_obj.area+ dpe_obj.area*no_of_mrrs + dac_obj.area*no_of_dacs+ adc_obj.area*no_of_adc + soa_obj.area*no_of_soas + pd_obj.area*no_of_pds
        total_conv_unit_area = conv_unit_area*conv_dpu_count 
        # print("MRR AREA", dpe_obj.area*no_of_mrrs)
        # print('HQNNA Conv Unit', conv_unit_area)
        # fc unit area
        weight_bank_mrrs = fc_dpe_count*fc_dpe_size
        input_bank_mrrs = fc_dpe_size
        no_of_mrrs = input_bank_mrrs+weight_bank_mrrs
        no_of_dacs = no_of_mrrs
        no_of_soas = 0
        no_of_pds = fc_dpe_count
        no_of_adc = fc_dpe_count  
        
        no_of_rn = 1
        fc_unit_area = no_of_rn*rn_obj.area+dpe_obj.area*no_of_mrrs + dac_obj.area*no_of_dacs + adc_obj.area*no_of_adc + soa_obj.area*no_of_soas + pd_obj.area*no_of_pds 
        total_fc_unit_area = fc_unit_area*fc_dpu_count
        # print("MRR AREA", dpe_obj.area*no_of_mrrs)
        
        # print('HQNNA FC Unit', total_fc_unit_area)
        total_area = total_conv_unit_area + total_fc_unit_area
        print("HQNNA Area", total_area, "mm2")
    elif vdp_type =="SCONNA":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_1b()
        adc_obj = ADC_8b()
        soa_obj = SOA()
        pd_obj = PD()
        BtoS_obj = BtoSLookUpTable()
        
        # sconna unit area
        osm_mrrs = dpe_count*dpe_size
        filter_mrrs = dpe_count*dpe_size
        no_of_mrrs = filter_mrrs+filter_mrrs
        no_of_dacs = osm_mrrs
        no_of_soas = 0
        no_of_pds = dpe_count
        no_of_adc = dpe_count 
        sconna_unit_area = dpe_obj.area*no_of_mrrs + dac_obj.area*no_of_dacs + adc_obj.area*no_of_adc  + pd_obj.area*no_of_pds + (BtoS_obj.area*dpe_count)/(32)
        total_sconna_unit_area = sconna_unit_area*dpu_count 
        total_area = total_sconna_unit_area
        print("SCONNA area", total_area, "mm2")
    elif vdp_type =="HSCONNA":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_1b()
        adc_obj = ADC_8b()
        soa_obj = SOA()
        pd_obj = PD()
        BtoS_obj = BToSConverter()
        
        
        # sconna unit area
        osm_bank_mrrs = dpe_count*dpe_size
        filter_mrrs = 4*dpe_count*dpe_size+dpe_count
        no_of_mrrs = filter_mrrs+osm_bank_mrrs
        no_of_dacs = osm_bank_mrrs
        no_of_soas = 0
        no_of_pds = dpe_count
        no_of_adc = dpe_count 
        hsconna_unit_area = dpe_obj.area*osm_bank_mrrs + dac_obj.area*no_of_dacs + adc_obj.area*no_of_adc + pd_obj.area*no_of_pds + BtoS_obj.area*dpe_count
        total_hsconna_unit_area = hsconna_unit_area*dpu_count 
        total_area = total_hsconna_unit_area
        print("HSCONNA area", total_area, "mm2")
    elif vdp_type =="ROBIN":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        act_dac_obj = DAC_1b()
        wgt_dac_obj = DAC_4b()
        adc_obj = ADC_16b()
        soa_obj = SOA()
        vcsel_obj = VCSEL()
        pd_obj = PD()
        
        # ROBIN unit area
        weight_bank_mrrs = dpe_count*dpe_size
        input_bank_mrrs = dpe_count*dpe_size
        no_of_mrrs = weight_bank_mrrs+input_bank_mrrs
        no_of_vcsel = dpe_count
        no_of_pds = dpe_count
        no_of_adc = dpe_count  
        no_of_rn = 1
        robin_unit_area = no_of_rn*rn_obj.area+dpe_obj.area*no_of_mrrs + wgt_dac_obj.area*weight_bank_mrrs + act_dac_obj.area*input_bank_mrrs + adc_obj.area*no_of_adc + vcsel_obj.area*no_of_vcsel+ pd_obj.area*no_of_pds
        # print('ROBIN Unit area', robin_unit_area)
        total_robin_unit_area = robin_unit_area*dpu_count 
        total_area = total_robin_unit_area
        print("ROBIN area", total_area, "mm2")
    elif vdp_type =="OXBNN":
        dpe_obj = MRR_DPE(conv_dpe_size,data_rate)
        rn_obj = RN(reduction_network_type)
        dac_obj = DAC_1b()
        adc_obj = ADC_8b()
        pd_obj = PD()
        # oxbnn unit area
        
        no_of_mrrs = dpe_count*dpe_size
        no_of_dacs = no_of_mrrs*2
        no_of_soas = 0
        no_of_pds = dpe_count
        no_of_adc = dpe_count 
        oxbnn_unit_area = dpe_obj.area*no_of_mrrs + dac_obj.area*no_of_dacs + adc_obj.area*no_of_adc  + pd_obj.area*no_of_pds 
        # print('OXBNN Unit area', oxbnn_unit_area)
        total_oxbnn_unit_area = oxbnn_unit_area*dpu_count 
        total_area = total_oxbnn_unit_area
        print("OXBNN area", total_area, "mm2")