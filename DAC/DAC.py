class DAC:
    def __init__(self, datarate = '1GHz'):
        self.no_of_parallel_requests = 1
        self.power = 0.0078e-3
        self.energy = 1.215e-12
        self.latency = 0.78e-9 # * from holylight 1 Cycle for DAC and clock at 1.28 GHz so latency 1/1.28 GHz
        self.area = 0.06 #mm2
        if datarate == '1GHz':
            self.power = 2e-3
            self.energy = 49.7e-9
            self.area = 0.014 #mm2
        # * Each MRR has a dedicated DAC so no need of queues, start time end time
        self.request_queue = None
        self.waiting_queue = None
        self.start_time = 0
        self.end_time = 0
        self.calls_count = 0