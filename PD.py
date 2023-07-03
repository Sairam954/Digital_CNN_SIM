class PD:
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 2.8 #mW
        self.latency = 5.8 #ps
        self.energy = self.power*self.latency
        self.area = 4e-5 #mm2
        # * A Low-Voltage Si-Ge Avalanche Photodiode for High-Speed and Energy Efficient Silicon Photonic Links

