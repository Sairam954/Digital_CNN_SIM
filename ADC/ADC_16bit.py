class ADC_16b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 62 #mW
        self.latency = 14 #ns
        self.energy = self.power*self.latency
        self.area = 0.55 #mm2
        # * A 16-bit 16-MS/s SAR ADC With On-Chip Calibration in 55-nm CMOS