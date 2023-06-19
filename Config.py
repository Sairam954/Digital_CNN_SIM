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
ELEMENT_SIZE = 'element_size'
ELEMENT_COUNT = 'element_count'
UNITS_COUNT = 'units_count'
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

TEST_WS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_IS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_OS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_ROS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_RWS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_RIS_S_TREE =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "S_Tree",     
}]

TEST_IS_PCA =    [{
    ELEMENT_SIZE: 50,    
    ELEMENT_COUNT: 50, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree",     
}]



AMM_WS_S_TREE =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "S_Tree",     
}]
AMM_IS_S_TREE=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree",     
}]

AMM_OS_S_TREE=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "S_Tree",     
}]
MAM_WS_S_TREE =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "S_Tree",     
}]
MAM_IS_S_TREE=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree",     
}]

MAM_OS_S_TREE=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "S_Tree",     
}]

AMM_WS_ST_Tree_Ac =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
AMM_IS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

AMM_OS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
MAM_WS_ST_Tree_Ac =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
MAM_IS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

MAM_OS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

AMM_WS_STIFT =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "STIFT",     
}]
AMM_IS_STIFT=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "STIFT",     
}]

AMM_OS_STIFT=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "STIFT",     
}]
MAM_WS_STIFT =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "STIFT",     
}]
MAM_IS_STIFT=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "STIFT",     
}]

MAM_OS_STIFT=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "STIFT",     
}]

AMM_WS_PCA =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "PCA",     
}]
AMM_IS_PCA=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA",     
}]

AMM_OS_PCA=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "PCA",     
}]
MAM_WS_PCA =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "PCA",     
}]
MAM_IS_PCA=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA",     
}]

MAM_OS_PCA=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "PCA",     
}]


AMM_RWS_S_TREE =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "S_Tree",     
}]
AMM_RIS_S_TREE=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "S_Tree",     
}]

AMM_ROS_S_TREE=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "S_Tree",     
}]
MAM_RWS_S_TREE =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "S_Tree",     
}]
MAM_RIS_S_TREE=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "S_Tree",     
}]

MAM_ROS_S_TREE=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "S_Tree",     
}]

AMM_RWS_ST_Tree_Ac =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
AMM_RIS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

AMM_ROS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
MAM_RWS_ST_Tree_Ac =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]
MAM_RIS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

MAM_ROS_ST_Tree_Ac=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac",     
}]

AMM_RWS_STIFT =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT",     
}]
AMM_RIS_STIFT=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT",     
}]

AMM_ROS_STIFT=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT",     
}]
MAM_RWS_STIFT =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT",     
}]
MAM_RIS_STIFT=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT",     
}]

MAM_ROS_STIFT=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT",     
}]

AMM_RWS_PCA =    [{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA",     
}]
AMM_RIS_PCA=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA",     
}]

AMM_ROS_PCA=[{
    ELEMENT_SIZE: 31,    
    ELEMENT_COUNT: 31, # number of multiplier    
    UNITS_COUNT: 98, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA",     
}]
MAM_RWS_PCA =    [{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA",     
}]
MAM_RIS_PCA=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA",     
}]

MAM_ROS_PCA=[{
    ELEMENT_SIZE: 44,    
    ELEMENT_COUNT: 44, # number of multiplier    
    UNITS_COUNT: 50, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA",     
}]
