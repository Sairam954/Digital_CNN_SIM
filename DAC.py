class DAC:
    def __init__(self, datarate = 1):
        self.no_of_parallel_requests = 1
        self.power = 0.0078e-3
        self.energy = 1.215e-12
        self.latency = 1/datarate 
        self.latency = self.latency*1e-9 # seconds  
        self.area = 0.06 #mm2
    
            
        # * Each MRR has a dedicated DAC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0