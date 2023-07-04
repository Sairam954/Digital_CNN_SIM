LAYER_TYPE = "name"
MODEL_NAME = "model_name"
KERNEL_DEPTH = "kernel_depth"
KERNEL_HEIGHT = "kernel_height"
KERNEL_WIDTH = "kernel_width"
TENSOR_COUNT = "tensor_count"
INPUT_SHAPE = "input_shape"
OUTPUT_SHAPE = "output_shape"
TENSOR_SHAPE = "tensor_shape"
INPUT_HEIGHT = "input_height"
INPUT_WIDTH = "input_width"
INPUT_DEPTH = "input_depth"
OUTPUT_HEIGHT = "output_height"
OUTPUT_WIDTH = "output_width"
OUTPUT_DEPTH = "output_depth"
ELEMENT_SIZE = "element_size"
ELEMENT_COUNT = "element_count"
CONV_ELEMENT_SIZE = "conv_element_size"
CONV_ELEMENT_COUNT = "conv_element_count"
FC_ELEMENT_SIZE = "fc_element_size"
FC_ELEMENT_COUNT = "fc_element_count"
CLUSTER_COUNT = "cluster_count"
UNITS_COUNT = 'units_count'
CONV_UNITS_COUNT = 'conv_units_count'
FC_UNITS_COUNT = 'fc_units_count'
RECONFIG = "reconfig"
ZERO = 0
LAST = -1 
ADDER = "adder"
POOL = "pool"
LAYERS_SUPPORTED = ["convolution","inner_product"]
ACC_TYPE ="accelerator_type" 
PRECISION = "precision"
BITRATE = "bitrate"
PCA = "pca"
DATAFLOW = "dataflow"
REDUCTION_TYPE = "reduction_type"
VDP_TYPE = "vdp_type"
NAME = "name"
BATCH_SIZE = "batch_size"
ACT_PRECISION = "act_precision"
WT_PRECISION = "wt_precision"

TEST_HQNNA =    [{
    ELEMENT_SIZE: 0,    
    ELEMENT_COUNT: 0,
    CONV_ELEMENT_SIZE : 20,   # number of multiplier   
    CONV_ELEMENT_COUNT : 20,
    FC_ELEMENT_SIZE : 50,   # number of multiplier   
    FC_ELEMENT_COUNT : 50,
    UNITS_COUNT: 100, 
    CONV_UNITS_COUNT: 100,
    FC_UNITS_COUNT: 200,
    RECONFIG: [], 
    VDP_TYPE: "HQNNA", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 1 ,   
}]
