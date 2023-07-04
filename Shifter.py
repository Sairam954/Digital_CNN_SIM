class Shifter:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 0.20 #mW
        self.latency = 0.78 #ps
        self.energy = self.power*self.latency
        self.area = 6.4e-3 #mm2
        # * Ultrahigh Speed Reconfigurable Logic Operations Based on Single Semiconductor Optical Amplifier