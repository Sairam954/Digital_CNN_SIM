import numpy as np
from Exceptions.AcceleratorExceptions import VDPElementException, VDPException

from Hardware.Multiplier import Multiplier
from constants import *
import logging as logging

logger = logging.getLogger("__MRR_VDP__")
logger.setLevel(logging.INFO)


class MultiplierNetwork():
    """[summary]

    Args:
        VDP ([type]): [description]

    Raises:
        VDPException: [description]
        VDPElementException: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, vdp_type,  supported_layer_list=[], BR=1) -> None:
        self.vdp_type = vdp_type
        self.start_time = 0
        self.end_time = 0
        self.multipier_network_list = []
        self.calls_count = 0
        self.br = BR*1e9  # in Hz
        self.layer_supported = LAYERS_SUPPORTED


    def does_support_layer(self, layer_name):
        if layer_name in self.layer_supported:
            return True
        else:
            return False

    def set_vdp_latency(self) -> float:
        self.latency = 1/(self.br) # todo : add the latency calculation of multiplier network, should be clock cycle 
        return self.latency
    def add_multiplier(self, multiplier):
        self.multipier_network_list.append(multiplier)

    def get_multiplier_count(self) -> int:
        return len(self.multipier_network_list)

    # * Aggregation using mux and splitting user splitter MRR are not including the calculations
    def get_utilized_and_idle_multipliers_convo(self, element_convo_count, kernel_size, element_size):
        total_multipliers = self.get_multiplier_count()
        utilized_rings = kernel_size
        idle_rings = total_multipliers - utilized_rings
        return {"utilized_rings": utilized_rings, "idle_rings": idle_rings}

    # todo utilization funtion for fc now its directly in controller logic
    def perform_convo_count(self,dot_product_size):
            """[The method returns for the given kernel_size = 9,25 etc how many such operations can this vdp element perform ]

        Args:
            kernel_size ([type]): [description]

        Raises:
            VDPElementException: [description]

        Returns:
            [type]: [description]
        """
    
            if dot_product_size>self.get_multiplier_count():
                raise VDPElementException("Cannot Map the Kernel Size to VDP directly please decompose")
            else:
                return self.get_multiplier_count()
    
    
    def __str__(self) -> str:

        return " Elements Count :"+str(self.get_multiplier_count()) + " Element Type "+self.vdp_type.__str__()

    def reset(self):
        self.start_time = 0
        self.end_time = 0
