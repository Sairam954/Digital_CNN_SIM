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
CLUSTER_COUNT = "cluster_count"
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
AMW_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 36,    
    ELEMENT_COUNT: 36, # number of multiplier    
    UNITS_COUNT: 207, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 36 ,   
}]


MAW_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 43,    
    ELEMENT_COUNT: 43, # number of multiplier    
    UNITS_COUNT: 280, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAW",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 43 ,   
}]


HEANA_IS_PCA_LS =    [{
    ELEMENT_SIZE: 73,    
    ELEMENT_COUNT: 73, # number of multiplier    
    UNITS_COUNT: 52, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "HEANA",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 73 ,   
}]

AMW5_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 17,    
    ELEMENT_COUNT: 17, # number of multiplier    
    UNITS_COUNT: 900, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMW5",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 5 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 17 ,   
}]


MAW5_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 21,    
    ELEMENT_COUNT: 21, # number of multiplier    
    UNITS_COUNT: 1100, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAW5",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 5 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 21 ,   
}]


HEANA5_IS_PCA_LS =    [{
    ELEMENT_SIZE: 39,    
    ELEMENT_COUNT: 39, # number of multiplier    
    UNITS_COUNT: 180, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "HEANA5",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 5 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 39 ,   
}]

AMW10_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 12,    
    ELEMENT_COUNT: 12, # number of multiplier    
    UNITS_COUNT: 1610, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMW10",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 12 ,   
}]


MAW10_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 15,    
    ELEMENT_COUNT: 15, # number of multiplier    
    UNITS_COUNT: 1950, 
    RECONFIG: [], 
    VDP_TYPE: "MAM", 
    NAME: "MAW10",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "S_Tree", 
    CLUSTER_COUNT : 15 ,   
}]


HEANA10_IS_PCA_LS =    [{
    ELEMENT_SIZE: 28,    
    ELEMENT_COUNT: 28, # number of multiplier    
    UNITS_COUNT: 320, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "HEANA10",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 28 ,   
}]