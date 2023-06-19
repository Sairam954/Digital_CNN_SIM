class MRR_DPE:
    
    def __init__(self, X, dr):
        self.num_of_weight_bank_mrr = X
        self.pitch = 5 # um
        self.radius = 4.55 # um
        self.waveguide_n = 3.48
        self.dr = dr # datarate GHz
        self.input_actuation_latency = 1/dr # ns
        self.thermo_optic_tuning_latency = 4 # um
        self.input_actuation_power = 1.6e-6 # W electro optic tuning
        self.weight_actuation_power = 1.375e-3 # W thermo optic tuning
        self.area = 3.14*(self.radius**2)*1e-6 # mm^2
        
    def get_prop_latency(self):
        path_distance = self.radius*2*3.143 + self.num_of_weight_bank_mrr*(self.radius*2+self.pitch)
        prop_latency = (path_distance*1e-6/(3*1e8)) # m/m/s = s 
        return prop_latency      