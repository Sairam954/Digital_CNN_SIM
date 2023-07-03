class SOA:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 2.2 #mW
        self.latency = 0.3 #ns
        self.energy = self.power*self.latency
        self.area = 1 #mm2
        # * Ultrahigh Speed Reconfigurable Logic Operations Based on Single Semiconductor Optical Amplifier