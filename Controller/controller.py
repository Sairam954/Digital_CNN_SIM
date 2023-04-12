from Exceptions.AcceleratorExceptions import VDPElementException
import math
import numpy as np

class Controller:
    
    # Todo need to intialize properties related to the controller 
    def __init__(self):
        self.utilized_rings = np.uint64(0)
        self.idle_rings = np.uint64(0)
    
    def get_channel_latency(self,accelerator,channels,convolutions_per_channel,kernel_size):
        total_latency = 0
        for channel in range(channels):
            total_latency += self.get_convolution_latency(accelerator,convolutions_per_channel,kernel_size)
            print(total_latency)
        return total_latency
    
    def get_partial_convolution_latency(self,clock,clock_increment,accelerator,partial_convolutions,kernel_size):
        """ This method is to perform convolution of vdp unit which cannot perform a single convolution operation even after decomposition
            This method thus calculate the latency by updating the clock, the partial convolution input will be always have kernel size equivalent to kernel
            size equivalent to vdp unit size. 

        Args:
            clock ([type]): [description]
            clock_increment ([type]): [description]
            accelerator ([type]): [description]
            partial_convolutions ([type]): [description]
            kernel_size ([type]): [description]

        Returns:
            [type]: [description]
        """
        ZERO = 0
        LAST = -1 
        ADDER = "adder"
        UTILIZED_RINGS = "utilized_rings"
        IDLE_RINGS = "idle_rings"
        cycle = 0
        completed_layer = False
        # print("Partial Sum Convolution")
        while clock>=0:
            vdp_no = 0
            for vdp in accelerator.vdp_units_list:
                # print('VDP Number ', vdp_no)
                # print("VDP End Time: ", vdp.end_time)
                # print(" Supported Layers ",vdp.layer_supported)
                if vdp.end_time <= clock:
                    # print("VDP unit Available Vdp No ", vdp_no)
                    vdp.start_time = clock
                    vdp.end_time = clock+vdp.latency
                    vdp.calls_count +=1
                    vdpelement = vdp.vdp_element_list[ZERO]
                    vdp_convo_count = 0
                    
                    # * element convo count contains the value of number of kernel size convo performed by vdp for reconfig it can be greater than one
                    try:
                        element_convo_count = vdpelement.perform_convo_count(kernel_size)
                    # * This situation arises in hybrid model where VDP element size is not constant
                    #* IF the element size are say 15 and 10. Due to large VDP size partial sum is needed then it is decomposed if it is decomposed based on 15
                    #* then partial sum is divided according to the largere and the smaller vdo element cannot perform this operation.
                    #* Have tried to 
                    except Exception as error:
                        # print("This VDP cannot process this element size")
                        vdp.end_time = clock
                        element_convo_count = 0
                    vdp_convo_count = element_convo_count*vdp.get_element_count()
                    # print("VDP Convolution Count",vdp_convo_count)
                    partial_convolutions = partial_convolutions-vdp_convo_count
                    # *  AMM has array of Weight and Input so 2 + element size to represent the rings in the input WDM mux
                    vdp_mrr_utiliz = vdp.get_utilized_idle_rings_convo(element_convo_count,kernel_size,vdpelement.element_size)
                    self.utilized_rings += vdp_mrr_utiliz[UTILIZED_RINGS]
                    self.idle_rings += vdp_mrr_utiliz[IDLE_RINGS]
                        # print("Utilized Rings :", self.utilized_rings)
                        # print("Idle Rings :",self.idle_rings)   
                    # print("--------------Partial Convolutions Left-------------------", partial_convolutions)  
                    if partial_convolutions <= 0:
                        completed_layer=True
                        # print("************Partial convolutions Completed****************",partial_convolutions)
                        break  
                else:
                    # print("VDP Unit Unavailable VDP NO:",vdp_no) 
                    pass
                if completed_layer:
                    break
                vdp_no += 1 
            if completed_layer:
                break
            cycle+=1
            # print('partial cycle', cycle)
            clock=clock+clock_increment  
            # print('Clock', clock)
        return clock,accelerator
    def  get_latency(self, accelerator, convolutions, dot_product_size, output_col, dataflow='WS', pca=False):
        """[  Function has to give the latency taken by the given accelerator to perform stated counvolutions with mentioned kernel size
        ]

        Args:
            accelerator ([Hardware.Accelerator]): [Accelerator for performing the convolutions]
            convolution_count ([type]): [No of convolutions to be performed by the accelerator]
            kernel_size ([type]): [size of the convolution]
            output col: Number of columns in the toepplitz matrix
            dataflow: Dataflow of the accelerator
            
        Returns:
            [float]: [returns the latency required by the accelerator to perform all the convolutions]
        """
        ELEMENT_SIZE = 'element_size'
        ELEMENT_COUNT = 'element_count'
        UNITS_COUNT = 'units_count'
        RECONFIG = "reconfig"
        ZERO = 0
        LAST = -1 
        ADDER = "adder"
        UTILIZED_MULTIPLIER = "utilized_multipliers"
        IDLE_MULTIPLIER = "idle_multipliers"
        
        cache_size = accelerator.cache_size
        clock = 0
        clock_increment = accelerator.vdp_units_list[ZERO].latency
        completed_layer = False
        cycle = 0
        
        while clock>=0:
            vdp_no = 0
            partial_sum_list = []
            accelerator.pheripherals[ADDER].controller(clock)
            for vdp in accelerator.vdp_units_list:
                if vdp.end_time <= clock:
                    vdp.start_time = clock
                    vdp.end_time = clock+vdp.latency
                    vdp.calls_count +=1
                    multiplier_network = vdp
                    vdp_convo_count = 0
                    try:
                        # print("Dot Product Size",dot_product_size)
                        element_convo_count = multiplier_network.perform_convo_count(dot_product_size)
                        performed_dot_product = 1
                        # print("Element VDP Count",element_convo_count)
                        convolutions = convolutions-performed_dot_product
                        vdp_cache_reads = performed_dot_product*multiplier_network.get_multiplier_count()*2 # OS , 2 to account for both inputs and weights
                        accelerator.cache_reads += vdp_cache_reads
                        accelerator.cache_writes +=  math.ceil((vdp_cache_reads)/(cache_size)) # TODO need to check if it is correct
                    except(VDPElementException):
                        # print("Need to Decompose Dot Product")
                        
                        sliced_dot_product_size = multiplier_network.get_multiplier_count()
                        # print("Decomposed dot product Size",sliced_dot_product_size)
                        folds = math.ceil(dot_product_size/sliced_dot_product_size)
                        # print("Folds Required", folds)
                        element_convo_count = multiplier_network.perform_convo_count(sliced_dot_product_size)
                        # print("VDPE Convolution Count ", element_convo_count)
                        vdp.end_time += vdp.latency*folds
                        partial_sum_latency = 0 
                        accelerator.psum_reads += folds # TODO need to check if it is correct
                        accelerator.psum_writes += folds
                        fold_idx = 0
                        # while folds>0:
                        #     # print("Processing Fold", fold_idx)
                        psums = multiplier_network.get_multiplier_count()*folds
                        partial_sum_latency += accelerator.pheripherals[ADDER].get_request_latency(psums,folds)
                        # print('partial_sum_latency ', partial_sum_latency)
                        # print("Multiplication Network Latency ", vdp.latency*folds)
                        vdp_cache_reads = multiplier_network.get_multiplier_count()*2*folds # OS , 2 to account for both inputs and weights
                        accelerator.cache_reads += vdp_cache_reads*folds
                        accelerator.cache_writes +=  math.ceil((vdp_cache_reads)/(cache_size)) # TODO need to check if it is correct
                        # folds-=1 
                        # fold_idx+=1
                        vdp.calls_count +=folds
                        performed_dot_product = 1
                        vdp.end_time += partial_sum_latency
                        # print("Utilized Rings :", self.utilized_rings)
                        # print("Idle Rings :",self.idle_rings)
                        convolutions = convolutions-performed_dot_product
                        # print("Remaining Convolutions", convolutions)
                    if convolutions <= 0:
                        completed_layer=True
                        # print("************Convolutions Completed****************",convolutions)
                        break   
                    
                    # print("Convolutions Left :", convolutions)
                   
                else:
                    pass
                    # print("VDP Unit Unavailable VDP NO:",vdp_no)
               
                if completed_layer:
                    break
                vdp_no += 1
            # print("Convolutions Left :", convolutions)
           
            if completed_layer:
                break
            cycle+=1
            #! Doing this to reduce the simulation time, the below clock update should not be done when accelerator has more than one VDP
            # clock=clock+clock_increment
            clock = accelerator.vdp_units_list[LAST].end_time+clock_increment
        # print('Conv Latency', clock)
        
        for vdp in accelerator.vdp_units_list:
            if vdp.end_time > clock:
                clock = vdp.end_time
        # print('PSum Latency', accelerator.pheripherals[ADDER].get_waiting_list_latency())   
        clock = clock + accelerator.pheripherals[ADDER].get_waiting_list_latency()
        # print('Clock', clock)
        accelerator.pheripherals[ADDER].reset()
        return clock 
    
    
