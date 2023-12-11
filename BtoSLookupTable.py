class BtoSLookUpTable:
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 2.38 #mW
        self.latency = 2.25 #ns
        self.energy = 0.25*1e3 #pJ
        self.area = 5.12 #mm2 
        # Cacti Simulation results ddr3 cache used for look up table analysis, size of the lookup table is calculate as 256*256 combinations and each combination of 512 bits, total bytes = 4194304