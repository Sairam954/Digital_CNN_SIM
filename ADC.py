class ADC():
    
    def __init__(self, datarate = 1):
        self.no_of_parallel_requests = 1
        self.latency = 1/datarate # nano seconds # * from holylight 1 Cycle for ADC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.latency = self.latency*1e-9 # seconds
        self.area = 0.014 #mm2
        if datarate == 1:
            self.power = 2e-3 #W
            self.energy = self.power*self.latency
            self.area = 0.002 #mm2
        elif datarate == 5:
            self.power = 11e-3
            self.energy = self.power*self.latency
            self.area = 0.021 #mm2
        elif datarate == 10: 
            self.power = 29e-3
            self.energy = self.power*self.latency
            self.area = 0.103 #mm2   
        # * Each MRR has a dedicated ADC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0