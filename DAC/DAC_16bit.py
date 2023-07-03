class DAC_16b:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 40e-3
        self.energy = 13.2e-12
        self.latency = 0.33e-9 
        self.area = 0.16 #mm2
        # * A 24.7 mW 65 nm CMOS SAR-Assisted CT ΔΣ Modulator With Second-Order Noise Coupling Achieving 45 MHz Bandwidth and 75.3 dB SNDR