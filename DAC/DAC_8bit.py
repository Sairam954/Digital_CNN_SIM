class DAC_8b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 3  #mW
        self.energy = 0.87 #pJ
        self.latency = 0.29 #ns
        self.area = 0.034 #mm2
        # * A 3 mW 6-bit 4 GS/s Subranging ADC With 2-bitSubrange-Dependent Embedded References