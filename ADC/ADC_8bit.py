class ADC_8b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 3.1e-3
        self.latency = 0.82e-9
        self.energy = self.power*self.latency
        self.area = 0.0015 #mm2
        # * A 16-bit 16-MS/s SAR ADC With On-Chip Calibration in 55-nm CMOS