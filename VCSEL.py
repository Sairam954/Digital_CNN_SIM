class VCSEL:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 0.66 #mW
        self.latency = 10 #ns
        self.energy = self.power*self.latency
        self.area = 1 #mm2 # TODO need to update this value from below paper
        # *“Efficient Hybrid Integration of Long-Wavelength VCSELs on Silicon Photonic Circuits