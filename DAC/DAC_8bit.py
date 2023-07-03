class DAC_8b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 3e-3
        self.energy = 0.87e-12
        self.latency = 0.29e-9 
        self.area = 0.034 #mm2
        # * A 3 mW 6-bit 4 GS/s Subranging ADC With 2-bitSubrange-Dependent Embedded References