class Multiplier:
    # todo: add the multiplier parameters like power, energy, latency, area
    def __init__(self):
        self.no_of_parallel_requests = 1
        self.power = 1.375e-3 
        self.energy = 5e-14  
        self.latency = None 
        self.area = 1001*1e-6  #! mm2 Took this value from stonne energy table text for multiplier, not sure about the units though
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0 