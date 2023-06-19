class VoltageAdder():
    
    def __init__(self):
        self.power = 0.382*1e-12
        self.energy = 0.0
        self.latency = 0.7932*1e-12 # * from holylight 1 Cycle for ADC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = 0.0