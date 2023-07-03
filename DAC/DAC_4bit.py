class DAC_4b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 0.45 #mW
        self.latency = 0.82 #ns
        self.energy = self.power*self.latency
        self.area = 0.0015 #mm2
        # * L. Kull et al., "A 3.1 mW 8b 1.2 GS/s Single-Channel Asynchronous SAR ADC With Alternate Comparators for Enhanced Speed in 32 nm Digital SOI CMOS," IEEE Journal of Solid-State Circuits, Dec 2013
        # * Assumptions made according to HQNNA paper

