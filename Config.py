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

TEST_RWS_S_TREE_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 1 ,   
}]

TEST_ROS_S_TREE_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 1 ,   
}]

TEST_RIS_S_TREE_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 1 ,   
}]

TEST_RWS_ST_TREE_AC_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_ROS_ST_TREE_AC_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_RIS_ST_TREE_AC_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_RWS_STIFT_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_ROS_STIFT_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_RIS_STIFT_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_RWS_PCA_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_ROS_PCA_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 1 ,   
}]

TEST_RIS_PCA_L1 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 1 ,   
}]


TEST_RWS_S_TREE_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 2,   
}]

TEST_ROS_S_TREE_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 2,   
}]

TEST_RIS_S_TREE_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 2,   
}]

TEST_RWS_ST_TREE_AC_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 2,   
}]

TEST_ROS_ST_TREE_AC_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 2,   
}]

TEST_RIS_ST_TREE_AC_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 2,   
}]

TEST_RWS_STIFT_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 2,   
}]

TEST_ROS_STIFT_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 2,   
}]

TEST_RIS_STIFT_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 2,   
}]

TEST_RWS_PCA_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 2,   
}]

TEST_ROS_PCA_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 2,   
}]

TEST_RIS_PCA_L2 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 2,   
}]


TEST_RWS_S_TREE_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 4,   
}]

TEST_ROS_S_TREE_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 4,   
}]

TEST_RIS_S_TREE_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 4,   
}]

TEST_RWS_ST_TREE_AC_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 4,   
}]

TEST_ROS_ST_TREE_AC_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 4,   
}]

TEST_RIS_ST_TREE_AC_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 4,   
}]

TEST_RWS_STIFT_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 4,   
}]

TEST_ROS_STIFT_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 4,   
}]

TEST_RIS_STIFT_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 4,   
}]

TEST_RWS_PCA_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 4,   
}]

TEST_ROS_PCA_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 4,   
}]

TEST_RIS_PCA_L4 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 4,   
}]


TEST_RWS_S_TREE_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 8,   
}]

TEST_ROS_S_TREE_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 8,   
}]

TEST_RIS_S_TREE_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 8,   
}]

TEST_RWS_ST_TREE_AC_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 8,   
}]

TEST_ROS_ST_TREE_AC_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 8,   
}]

TEST_RIS_ST_TREE_AC_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 8,   
}]

TEST_RWS_STIFT_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 8,   
}]

TEST_ROS_STIFT_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 8,   
}]

TEST_RIS_STIFT_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 8,   
}]

TEST_RWS_PCA_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 8,   
}]

TEST_ROS_PCA_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 8,   
}]

TEST_RIS_PCA_L8 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 8,   
}]
TEST_RWS_S_TREE_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 16 ,   
}]

TEST_ROS_S_TREE_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 16 ,   
}]

TEST_RIS_S_TREE_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 16 ,   
}]

TEST_RWS_ST_TREE_AC_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_ROS_ST_TREE_AC_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_RIS_ST_TREE_AC_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_RWS_STIFT_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_ROS_STIFT_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_RIS_STIFT_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_RWS_PCA_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_ROS_PCA_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 16 ,   
}]

TEST_RIS_PCA_L16 =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 16 ,   
}]


TEST_RWS_S_TREE_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40,   
}]

TEST_ROS_S_TREE_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40,   
}]

TEST_RIS_S_TREE_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40,   
}]

TEST_RWS_ST_TREE_AC_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40,   
}]

TEST_ROS_ST_TREE_AC_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40,   
}]

TEST_RIS_ST_TREE_AC_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40,   
}]

TEST_RWS_STIFT_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40,   
}]

TEST_ROS_STIFT_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40,   
}]

TEST_RIS_STIFT_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40,   
}]

TEST_RWS_PCA_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RWS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40,   
}]

TEST_ROS_PCA_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "ROS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40,   
}]

TEST_RIS_PCA_LM =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "RIS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40,   
}]


TEST_WS_S_TREE_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40 ,   
}]

TEST_OS_S_TREE_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40 ,   
}]

TEST_IS_S_TREE_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
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
    CLUSTER_COUNT : 40 ,   
}]

TEST_WS_ST_TREE_AC_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_OS_ST_TREE_AC_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_IS_ST_TREE_AC_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "ST_Tree_Ac", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_WS_STIFT_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_OS_STIFT_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_IS_STIFT_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "STIFT", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_WS_PCA_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "WS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_OS_PCA_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 10 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "OS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40 ,   
}]

TEST_IS_PCA_LS =    [{
    ELEMENT_SIZE: 27,    
    ELEMENT_COUNT: 32, # number of multiplier    
    UNITS_COUNT: 100, 
    RECONFIG: [], 
    VDP_TYPE: "AMM", 
    NAME: "AMM",  
    ACC_TYPE: "DIGITAL", 
    PRECISION: 1, 
    BITRATE: 1 , # GHz
    BATCH_SIZE: 1,  
    DATAFLOW: "IS",
    REDUCTION_TYPE: "PCA", 
    CLUSTER_COUNT : 40 ,   
}]