import math


class RN:
    
    def __init__(self, RN_TYPE='S_Tree'):
        self.reduction_type = RN_TYPE
        self.latency = 1/(800*1e6)  #s # STIFT paper clock speed is 800 MHz then each cycle will be 1/800 MHz
        if RN_TYPE == 'S_Tree':
            self.area = 0.172E-06
            self.power = 352E-03
        elif RN_TYPE == 'ST_Tree_Ac':
             self.area = 0.32E-06
             self.power = 654E-03
        elif RN_TYPE == 'STIFT':
             self.area = 0.273E-06
             self.power = 529E-03
        elif RN_TYPE == 'PCA':
             self.area = 0
             self.power = 0
    def get_reduction_latency(self, psums, folds):
             
        if self.reduction_type == "S_Tree":
            adder_level = math.log2(psums)
            number_of_clocks = (folds+adder_level)*math.log2(psums/folds)
            self.energy = 2.86888E-05
        elif self.reduction_type == "ST_Tree_Ac" or self.reduction_type == "STIFT":
            number_of_clocks = (folds)*math.log2(psums/folds)
            self.energy = 2.97013E-05
            if self.reduction_type == "ST_Tree_Ac":
                self.power = 654E-03
            else:
                self.power = 529E-03
        elif self.reduction_type == "PCA":
            number_of_clocks = 0  
            self.power = 0
            self.area = 0
            self.energy = 0
        return self.latency*number_of_clocks
        