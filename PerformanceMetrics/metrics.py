from cmath import pi
from Hardware.Accumulator_TIA import Accumulator_TIA
from Hardware.BtoS import BtoS
from Hardware.MRR import MRR
from Hardware.Serializer import Serializer
from Hardware.eDram import EDram
from Hardware.ADC import ADC
from Hardware.DAC import DAC
from Hardware.PD import PD
from Hardware.TIA import TIA
from Hardware.VCSEL import VCSEL
from Hardware.io_interface import IOInterface
from Hardware.bus import Bus
from Hardware.router import Router
from Hardware.Activation import Activation
from Hardware.Multiplier import Multiplier


class Metrics:

    def __init__(self):
        self.eDram = EDram()
        self.adc = ADC()
        self.dac = DAC()
        self.pd = PD()
        self.tia = TIA()
        self.mrr = MRR()
        self.multiplier = Multiplier()
        self.io_interface = IOInterface()
        self.bus = Bus()
        self.router = Router()
        self.activation = Activation()
        self.serializer = Serializer()
        self.accum_tia = Accumulator_TIA()
        self.b_to_s = BtoS()
        self.vcsel = VCSEL()
        self.laser_power_per_wavelength = 1.274274986e-3
        self.wall_plug_efficiency = 5  # 20%
        self.thermal_tuning_latency = 4000e-9
        self.photonic_adder = 1060+5.12+200 
        self.cache_latency = 0.1e-9
        

    def get_hardware_utilization(self, utilized_rings, idle_rings):

        return (utilized_rings/(utilized_rings+idle_rings))*100

    def get_dynamic_energy(self, accelerator, utilized_rings, reduction_type, total_latency):

        total_energy = 0
        # * For each vdp in accelerator the number of calls gives the number of times eDRam is called
        # * The dynamic energy of ADC, DAC , MRR = no of rings utilized*their eneergy
        # * PD and TIA energy = vdp calls * no of vdp elements in each VDP

        for vdp in accelerator.vdp_units_list:      # Number of loops equals number of cluster (Loop for each cluster)
            eDram_energy = vdp.calls_count*self.eDram.energy
            cache_read_energy = (accelerator.cache_reads + accelerator.psum_reads)*67.5e-15*4 # 0.044 pJ per bit
            cache_write_energy = (accelerator.cache_writes + accelerator.psum_writes)*67.5e-15*4 # 0.044 pJ per bit
            cache_energy = cache_read_energy + cache_write_energy
            multiplier_energy = self.multiplier.energy*vdp.calls_count*vdp.get_multiplier_count()
            
            # Obtain reduction network dynamic energy based on which type of network
            reduction_energy = 0
            #if reduction_type == "Photonic_RN":
            #    reduction_energy += 8.9012e-12*vdp.calls_count*vdp.get_multiplier_count()       # DTC Energy

            #print("eDram_energy: ", eDram_energy)
            #print("cache_energy: ", cache_energy)
            #print("multiplier: ", multiplier_energy)
            #print("reduction_energy: ", reduction_energy)
            
            #total_energy += eDram_energy+multiplier_energy+reduction_energy
            total_energy += eDram_energy+cache_energy+multiplier_energy+reduction_energy
            
            
        return total_energy

    def get_total_latency(self, latencylist, accelerator):
        total_latency = sum(latencylist)
        # +(accelerator.cache_reads + accelerator.psum_reads +accelerator.cache_writes + accelerator.psum_writes)*self.cache_latency
        return total_latency

    def get_static_power(self, accelerator, reduction_type):

        total_power = 0
        vdp_power = 0
        for vdp in accelerator.vdp_units_list:
            # * adding no of comb switches  to the vdp element
            num_of_multipliers = vdp.get_multiplier_count()
            multiplier_power = self.multiplier.power*num_of_multipliers
            # todo : add the power of the digital to pulse converter and other elements in the vdp 
            # add the reduction network power
                # print("VDP Power ", vdp_power)
            pheripheral_power_params = {}
            pheripheral_power_params['io'] = self.io_interface.power
            pheripheral_power_params['bus'] = self.bus.power
            pheripheral_power_params['eram'] = self.eDram.power
            pheripheral_power_params['router'] = self.router.power
            pheripheral_power_params['activation'] = self.activation.power

            # Obtain reduction network static power based on which type of network
            reduction_power = 0
            if reduction_type == "S_Tree":
                reduction_power = 0.352
            elif reduction_type == "ST_Tree_AC":
                reduction_power = 0.654
            elif reduction_type == "STIFT":
                reduction_power = 0.529
            elif reduction_type == "Photonic_RN":
                reduction_power = 2.412004181

            total_power += self.io_interface.power + self.activation.power + \
                self.router.power + self.bus.power + vdp_power + self.eDram.power + reduction_power
            # print("Total Power ", total_power)
            
        return total_power

    def get_total_area(self, unit_count, element_count):
        pitch = 5  # um
        radius = 4.55  # um
        S_A_area = 0.00003  # mm2
        eDram_area = 0.166  # mm2
        max_pool_area = 0.00024  # mm2
        sigmoid = 0.0006  # mm2
        router = 0.151  # mm2
        bus = 0.009  # mm2
        splitter = 0.005  # mm2
        pd = 1.40625 * 1e-5  # mm2
        adc = self.adc.area  # mm2
        dac = self.dac.area  # mm2
        io_interface = 0.0244  # mm2
        serializer = 5.9*1e-3  # mm2
        
        
        multiplier_area = self.multiplier.area*element_count
        total_cnn_units_area = unit_count*multiplier_area # todo add reduction network and other components area
        total_area = total_cnn_units_area + S_A_area + \
            eDram_area + sigmoid + router + bus + max_pool_area+io_interface
        
        
        return total_area
