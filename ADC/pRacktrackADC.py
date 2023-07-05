class PCMADC_4b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 0.2 #mW
        self.latency = 0.08 #ns
        self.energy = 50 #fJ
        self.area = 0.00069 #mm2
        # * A 16-bit 16-MS/s SAR ADC With On-Chip Calibration in 55-nm CMOS