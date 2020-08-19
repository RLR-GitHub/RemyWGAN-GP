"""
#===============================================================================
#===============================================================================
"""
from TF_Environment import TF_Environment
from wgangp_functions_tmp import WGANGP
#from wgangp_functions import WGANGP
from FileObject import FileObject

C_CH      = 3
IMG_SIZE  = 128
INPUT_DIM = ( IMG_SIZE, IMG_SIZE, C_CH )
OPTIMIZER = "adam"                          # { "adam", "rmsprop" }
filepath  = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/data/results/"
filepath  += str( IMG_SIZE ) + "/architecture_" + str( IMG_SIZE ) + ".txt"
"""
#===============================================================================
"""
if( __name__ == "__main__" ):

    file_object = FileObject( filepath = filepath )
    session = TF_Environment( gpu_fraction = 0.8 )
    session.run_session()

    gan = WGANGP(
        #=======================================================================
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 500,
        save_rate = 10,
        print_rate = 5,
        input_dim = INPUT_DIM,
        optimizer = OPTIMIZER,
        batch_size = 20,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces_augmented/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces_augmented/",

        #=======================================================================

        critic_activation = 'leaky_relu',       # { 'relu', 'leaky_relu' }
        critic_dropout_rate = None,             # ISSUES
        critic_learning_rate = 0.0005,          # { 0.0001, 0.00005 }
        critic_gaussian_noise = False,          # CHANGE BACK TO TRUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        generator_activation = 'leaky_relu',          # { 'relu', 'leaky_relu' }
        generator_dropout_rate = None,
        generator_learning_rate = 0.0005,       # { 0.0001, 0.00005 }
        generator_instance_norm = False,
        generator_layer_normalization = False,  # CHANGE BACK TO TRUE
        generator_batch_norm_momentum = None,   # REQUIRES LARGE DATASET

        #=======================================================================

        #critic_conv_filters     = [  64, 128, 256, 512 ],
        #critic_conv_strides     = [   2,   2,   2,   1 ],
        #critic_conv_kernel_size = [   4,   4,   4,   4 ],

        #critic_conv_filters     = [  32,  64, 128, 256 ],
        #critic_conv_strides     = [   1,   2,   2,   2 ],
        #critic_conv_kernel_size = [   8,   4,   4,   4 ],

        #critic_conv_filters     = [   64,  64,  64 ],
        #critic_conv_strides     = [    2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4 ],

        #critic_conv_filters     = [   64, 128, 256 ],
        #critic_conv_strides     = [    2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   6,   4 ],

        #critic_conv_filters     = [   64, 128, 256 ],
        #critic_conv_strides     = [    2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   8,   8 ],

        critic_conv_filters     = [   32,  64, 128, 256 ],
        critic_conv_strides     = [    2,   2,   2,   1 ],
        critic_conv_kernel_size = [    4,   4,   4,   4 ],

        #generator_initial_dense_layer_size = (  4,  4,  64 ),
        #generator_conv_filters     = [   64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64, C_CH ],

        generator_initial_dense_layer_size = (  4,  4,   64 ),
        generator_conv_filters     = [   64,  64,  64,  64,  64,  64,  C_CH ],
        generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,    7  ],
        generator_conv_strides     = [    2,   2,   2,   2,   2,   1,    1  ],
        generator_upsample         = [    1,   1,   1,   1,   1,   1,    1  ],
        generator_conv_type        = [   't', 't', 't', 't', 't', 't',  't' ], # 'c' for regular, 't' for transposed

        #generator_initial_dense_layer_size = (  8,  8,  128 ),
        #generator_conv_filters     = [  128, 128, 128,  C_CH ],
        #generator_conv_kernel_size = [    5,   5,   5,    5  ],
        #generator_conv_strides     = [    1,   1,   1,    1  ],
        #generator_upsample         = [    2,   2,   2,    2  ],
        #generator_conv_type        = [   'c', 'c', 'c',  'c' ], # 'c' for regular, 't' for transposed

        #generator_initial_dense_layer_size = (  8,  8,  512 ),
        #generator_conv_filters     = [  512, 256, 128,  64,  C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   5,    7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,    1  ],
        #generator_upsample         = [    2,   2,   2,   2,    1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c',  'c' ], # 'c' for regular, 't' for transposed

        #generator_initial_dense_layer_size = (  4,  4,   64 ),
        #generator_conv_filters     = [   64,  64,  64,  64,  64,  C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   5,    7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,    1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,    1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c',  'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [   64,  64,  64,  64,  64,  64, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   5,   5,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_kernel_size = [    3,   3,   3,   3,   5,   5,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_initial_dense_layer_size = (  4,  4,  32 ),
        #generator_conv_filters     = [   32,  32,  32,  32,  32,  32,  32,  32,  32,  32, C_CH ],

        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   3,   5,   5,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   1,   2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  64,  64,  64,  64,  64,  64,  64,  64,  64,  64, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   2,   1,   2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [ 512, 512, 256, 256, 128, 128,  64,  64,  32, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   1,   2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #=======================================================================

    )

    x_train = gan.load_dataset( limit = 400 )
    gan.generate_random_image()
    gan.clear_terminal()
    gan.clean_folders()

    #gan.print_model_summaries( critic = True, generator = True )
    strings_list_0 = gan.print_model_layers( architecture = "parameters" )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )

    file_object.write2file( line_list = strings_list_0, open_file =  True, close_file = False )
    file_object.write2file( line_list = strings_list_1, open_file = False, close_file = False )
    file_object.write2file( line_list = strings_list_2, open_file = False, close_file =  True )

    gan.graph_update_rates( epoch_arr = [ 100, 250, 2000, 10000 ], update_rate = [ 25, 50, 100, 250 ] )

    if( OPTIMIZER == "adam" ): gan.lr_reductions( epoch_arr = [ 10000, 20000, 500000 ], lr_arr = [ 0.0004, 0.00025, 0.0001 ] )
    #elif( OPTIMIZER == "rmsprop" ): gan.lr_reductions( epoch_arr = [ 1000000 ], lr_arr = [ 0.00001 ] ) # NO NEED TO CHANGE LR FOR RMSPROP

    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 5 )
    session.end_session()
"""
#===============================================================================
#===============================================================================
"""





"""
        #=======================================================================

        #generator_initial_dense_layer_size = (  8,  8,  256 ),
        #generator_conv_filters     = [  256, 128, 128,  64,  64,  32,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   1,   2,   1,   2,   1,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #critic_conv_filters     = [  64,  64, 128, 128, 256 ],
        #critic_conv_strides     = [   1,   2,   1,   2,   1 ],
        #critic_conv_kernel_size = [   4,   4,   4,   4,   4 ],

        #generator_initial_dense_layer_size = (  8,  8,  256 ),

        #generator_conv_filters     = [  256, 128, 128,  64,  64,  32,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128, 128,  64,  64,  32,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   1,   2,   1,   2,   1,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed


        #generator_initial_dense_layer_size = (  8,  8,  256 ),
        #generator_conv_filters     = [ 256, 128,  64,  32, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   2,   2,   2,   2,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed


        #=======================================================================
"""



"""

        #=======================================================================
        z_dim = 128,
        img_size = IMG_SIZE,
        gif_rate = 500,
        save_rate = 10,
        print_rate = 5,
        input_dim = INPUT_DIM,
        optimizer = OPTIMIZER,
        batch_size = 5,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces_augmented/",
        #=======================================================================
        critic_activation = 'leaky_relu',       # { 'relu', 'leaky_relu' }
        critic_dropout_rate = None,             # ISSUES
        critic_learning_rate = 0.0005,          # { 0.0001, 0.00005 }
        critic_gaussian_noise = True,          # CHANGE BACK TO TRUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        generator_activation = 'relu',          # { 'relu', 'leaky_relu' }
        generator_dropout_rate = None,
        generator_learning_rate = 0.0005,       # { 0.0001, 0.00005 }
        generator_instance_norm = True,
        generator_layer_normalization = False,  # CHANGE BACK TO TRUE
        generator_batch_norm_momentum = None,   # REQUIRES LARGE DATASET
        #=======================================================================
        #critic_conv_filters     = [    64, 128, 256, 512  ],
        #critic_conv_strides     = [     2,   2,   2,   1  ],
        #critic_conv_kernel_size = [     4,   4,   4,   4  ],

        #critic_conv_filters     = [  32,  32,  64, 128, 256  ],
        #critic_conv_strides     = [   1,   2,   2,   2,   2  ],
        #critic_conv_kernel_size = [   8,   4,   4,   4,   4  ],

        #critic_conv_filters     = [  32,  64, 128, 256  ],
        #critic_conv_strides     = [   2,   2,   2,   1  ],
        #critic_conv_kernel_size = [  16,   8,   4,   4  ],

        #critic_conv_filters     = [  32,  64, 128 ],
        #critic_conv_strides     = [   2,   2,   2 ],
        #critic_conv_kernel_size = [  16,   8,   4 ],

        #critic_conv_filters     = [  64,  64, 128, 256 ],
        #critic_conv_strides     = [   1,   2,   2,   2 ],
        #critic_conv_kernel_size = [   8,   4,   4,   4 ],

        #critic_conv_filters     = [  64, 128, 256, 512 ],
        #critic_conv_strides     = [   2,   2,   2,   1 ],
        #critic_conv_kernel_size = [   4,   4,   4,   4 ],

        critic_conv_filters     = [  64,  64, 128, 128, 256 ],
        critic_conv_strides     = [   1,   2,   1,   2,   1 ],
        critic_conv_kernel_size = [   4,   4,   4,   4,   4 ],

        generator_initial_dense_layer_size = (  8,  8,  256 ),

        #generator_conv_filters     = [  256, 128, 128,  64,  64,  32,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        generator_conv_filters     = [  256, 128, 128,  64,  64,  32,  32, C_CH ],
        generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   7  ],
        generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1  ],
        generator_upsample         = [    2,   1,   2,   1,   2,   1,   2,   1  ],
        generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128, 128,  64,  64,  32,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   1,   2,   1,   2,   1,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed


        #generator_initial_dense_layer_size = (  8,  8,  256 ),
        #generator_conv_filters     = [ 256, 128,  64,  32, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   2,   2,   2,   2,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [ 256, 128,  64,  32,  32,  16, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   2,   1,   1  ],
        #generator_upsample         = [   2,   2,   2,   2,   1,   2,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128,  64,  32,  16,   8, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   5,   7,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,   1,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  512, 256, 128,  64,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  256, 256, 128,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   1,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  128, 128, 128, 128, 128, C_CH  ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   6,    7  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1,    1  ],
        #generator_upsample         = [    2,   2,   2,   2,   2,    1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c',  'c' ], # 'c' for regular, 't' for transposed

        #=======================================================================

"""

















"""
    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        gif = True,
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 1000,
        save_rate = 25,
        input_dim = INPUT_DIM,
        optimizer = "rmsprop",                      # { "adam", "rmsprop" }
        batch_size = 15,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces/",#"_augmented/",

        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces_augmented/",
        #===================================================================
        # BATCH_NORMALIZATION_MOMENTUM, DROPOUT & LEARNING RATES
        critic_batch_norm_momentum = None, # NEVER USE FOR WGAN-GP
        critic_learning_rate = 0.00075,
        critic_dropout_rate = None, # ISSUES

        generator_layer_normalization = True,
        generator_batch_norm_momentum = None,
        generator_learning_rate = 0.00075,
        generator_dropout_rate = None,
        #===================================================================
        # CRITIC

        #critic_conv_filters     = [   32,  64,  64,  64,  64 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   4,   4,   4,   4 ],

        critic_conv_filters     = [   128, 128, 128, 128 ],
        critic_conv_strides     = [     2,   2,   2,   2 ],
        critic_conv_kernel_size = [     8,   8,   8,   8 ],

        #critic_conv_filters     = [   64, 128, 128, 128, 128 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   64,  64,  64,  64,  64 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   32,  32,  32,  32,  32 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   16,  32,  32,  32,  32,  32 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   6,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   16,  32,  32,  32,  32,  32 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   32,  64,  64,  64,  64 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    8,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   32,  64, 128, 256 ],
        #critic_conv_strides     = [    2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4 ],
        #===================================================================
        # GENERATOR
        #generator_initial_dense_layer_size = (  4,  4, 256 ),
        generator_initial_dense_layer_size = (  4,  4, 256 ),

        generator_conv_filters     = [  128, 128, 128, 128, C_CH  ],
        generator_conv_kernel_size = [    4,   4,   4,   4,    4  ],
        generator_conv_strides     = [    2,   2,   2,   2,    2  ],
        generator_upsample         = [    1,   1,   1,   1,    1  ],
        generator_conv_type        = [   't', 't', 't', 't',  't' ], # 'c' for regular, 't' for transposed
        #generator_initial_dense_layer_size = (  8,  8, 256 ),

        #generator_conv_type        = "regular", # "regular"
        #generator_conv_filters     = [  128, 128, 128, 128, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    1,   1,   1,   1,    1 ],
        #generator_upsample         = [    2,   2,   2,   2,    1 ],

        #generator_conv_filters     = [  128, 128, 128, 128,  64, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    1 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    2,   2,   2,   2,   2,    2 ],

        #generator_conv_filters     = [  128, 128, 128, 128, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,    1 ],
        #generator_conv_strides     = [    2,   2,   2,   2,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,    1 ],


        #generator_conv_filters     = [  128, 128, 128, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,    2 ],
        #generator_upsample         = [    1,   1,   1,    1 ],

        #generator_conv_filters     = [   64,  64,  64,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   8,    8 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   2,    1 ],

        #generator_conv_filters     = [   64,  64,  64,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],

        #generator_conv_filters     = [   32,  32,  32,  32,  C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,     4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,     1 ],
        #generator_upsample         = [    1,   1,   1,   1,     1 ],

        #generator_conv_filters     = [   32,  32,  32,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],

        #generator_conv_filters     = [   32,  32,  32,  32,  32,  32, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,   1,    1 ],

        #generator_conv_filters     = [   64,  64,  64,  64,  64,  32, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,   1,    1 ],


        #generator_conv_filters     = [  256, 128,  64,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],
        #===================================================================
    )
"""
"""
    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        gif = True,
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 1000,
        save_rate = 50,
        input_dim = INPUT_DIM,
        optimizer = "rmsprop",                      # { "adam", "rmsprop" }
        batch_size = 10,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best/".             #"_augmented/",    #"best_200_faces/,"   #"_augmented/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces_augmented/",
        #===================================================================
        # BATCH_NORMALIZATION_MOMENTUM, DROPOUT & LEARNING RATES
        critic_batch_norm_momentum = None, # NEVER USE FOR WGAN-GP
        critic_learning_rate = 0.0005,# ~ 0.0002 is recommended
        critic_dropout_rate = None, # ISSUES

        generator_layer_normalization = True,
        generator_batch_norm_momentum = None,
        generator_learning_rate = 0.0005, # around 0.0002 is recommended
        generator_dropout_rate = None, #0.0001, #0.0001,
        #===================================================================
        # CRITIC

        critic_conv_filters     = [   64,  64,  64,  64,  64 ],
        critic_conv_strides     = [    2,   2,   2,   2,   2 ],
        critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   32,  64, 128, 256 ],
        #critic_conv_strides     = [    2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4 ],
        #===================================================================
        # GENERATOR
        generator_initial_dense_layer_size = (  8,  8, 128 ),

        generator_conv_filters     = [  128, 128, 128, 128,  C_CH ],
        generator_conv_kernel_size = [    4,   4,   4,   4,     4 ],
        generator_conv_strides     = [    2,   2,   2,   2,     1 ],
        generator_upsample         = [    1,   1,   1,   1,     1 ],

        #generator_conv_filters     = [  256, 128,  64,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],
        #===================================================================
    )

    x_train = gan.load_dataset( limit = 100 )
    gan.generate_random_image()
    gan.clear_terminal()
    gan.clean_folders()

    #gan.print_model_summaries( critic = True, generator = True )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )
    file_object.write2file( line_list = strings_list_1, open_file = True )
    file_object.write2file( line_list = strings_list_2, close_file = True )

    gan.lr_reductions( epoch_arr = [ 10000, 50000, 100000 ], lr_arr = [ 0.00025, 0.0001, 0.00005 ] )
    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 1 )
    session.end_session()

"""
"""
    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        gif = True,
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 1000,
        save_rate = 100,
        input_dim = INPUT_DIM,
        optimizer = "rmsprop",#"adam", # { "adam", "rmsprop" }
        batch_size = 10,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_augmented/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces_augmented/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_100_faces_augmented/",

        #===================================================================
        # BATCH_NORMALIZATION_MOMENTUM, DROPOUT & LEARNING RATES
        critic_batch_norm_momentum = None, # NEVER USE FOR WGAN-GP
        critic_learning_rate = 0.001,# ~ 0.0002 is recommended
        critic_dropout_rate = None, # ISSUES

        generator_layer_normalization = True,
        generator_batch_norm_momentum = None,
        generator_learning_rate = 0.001, # around 0.0002 is recommended
        generator_dropout_rate = 0.0001, #0.0001,
        #===================================================================
        # FILTERS, STRIDES, LAYER & KERNEL SIZES

        # critic (IN:128)       (OUT: 64,  32,  16,   8,   4 )
        #critic_conv_filters     = [   64,  64,  64,  64,  64 ],
        #critic_conv_strides     = [    2,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        critic_conv_filters     = [   64,  64,  64,  64,  64 ],
        critic_conv_strides     = [    2,   2,   2,   2,   2 ],
        critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        #critic_conv_filters     = [   32,  64, 128, 256 ],
        #critic_conv_strides     = [    2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4 ],

        #critic_conv_filters     = [   16,  32,  64, 128, 256 ],
        #critic_conv_strides     = [    1,   2,   2,   2,   2 ],
        #critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        # GENERATOR (IN:4)         (OUT:  8,  16,  32,  64, 128, 128 )
        generator_initial_dense_layer_size = (  8,  8, 128 ),
        #generator_initial_dense_layer_size = (  8,  8, 512 ),

        #generator_conv_filters     = [  256, 128,  64,  32,  16, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],

        #generator_conv_filters     = [  256, 128,  64,  32,  C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,     4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,     1 ],
        #generator_upsample         = [    1,   1,   1,   1,     1 ],

        generator_conv_filters     = [  128, 128, 128, 128,  C_CH ],
        generator_conv_kernel_size = [    4,   4,   4,   4,     4 ],
        generator_conv_strides     = [    2,   2,   2,   2,     1 ],
        generator_upsample         = [    1,   1,   1,   1,     1 ],

        #generator_conv_filters     = [  128, 128, 128, 128,  64, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [    2,   2,   2,   2,   1,    1 ],
        #generator_upsample         = [    1,   1,   1,   1,   1,    1 ],
        #===================================================================
    )

    x_train = gan.load_dataset( limit = 200 )
    gan.generate_random_image()
    gan.clear_terminal()
    gan.clean_folders()

    #gan.print_model_summaries( critic = True, generator = True )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )

    file_object.write2file( line_list = strings_list_1, open_file = True )
    file_object.write2file( line_list = strings_list_2, close_file = True )

    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 1 )

    session.end_session()
"""
"""
    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 500,
        save_rate = 250,
        input_dim = INPUT_DIM,
        optimizer = "adam",
        batch_size = 10,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_augmented/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces_augmented/",
        #===================================================================
        # BATCH_NORMALIZATION_MOMENTUM, DROPOUT & LEARNING RATES
        critic_batch_norm_momentum = None, # NEVER USE FOR WGAN-GP
        critic_learning_rate = 0.0002,# around 0.0002 is recommended
        critic_dropout_rate = None, # ISSUES

        generator_batch_norm_momentum = 0.9,
        generator_learning_rate = 0.0005, # around 0.0002 is recommended
        generator_dropout_rate = 0.0001,
        #===================================================================
        # FILTERS, STRIDES, LAYER & KERNEL SIZES

        # critic (IN:128)       (OUT: 64,  32,  16,   8,   4 )
        critic_conv_filters     = [   32,  64, 128, 256, 512 ],
        critic_conv_strides     = [    2,   2,   2,   2,   2 ],
        critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        # GENERATOR (IN:4)         (OUT:  8,  16,  32,  64, 128, 128 )
        generator_initial_dense_layer_size = ( 4, 4, 512 ),

        generator_conv_filters     = [  256, 128,  64,  32,  16, C_CH ],
        generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        generator_conv_strides     = [    1,   1,   1,   1,   1,    1 ],
        generator_upsample         = [    2,   2,   2,   2,   2,    1 ],# if not '2', uses Conv2DTranspose
        #===================================================================
    )

    x_train = gan.load_dataset( limit = 50 )
    gan.generate_random_image()
    #gan.clear_terminal()
    gan.clean_folders()

    gan.print_model_summaries( critic = True, generator = True )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )

    file_object.write2file( line_list = strings_list_1, open_file = True )
    file_object.write2file( line_list = strings_list_2, close_file = True )

    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 3 )

    session.end_session()
"""
"""

    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 500,
        save_rate = 250,
        input_dim = INPUT_DIM,
        optimizer = "adam",
        batch_size = 10,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_augmented/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces_augmented/",
        #===================================================================
        # BATCH_NORMALIZATION_MOMENTUM, DROPOUT & LEARNING RATES
        critic_batch_norm_momentum = None, # NEVER USE FOR WGAN-GP
        critic_learning_rate = 0.0001,# around 0.0002 is recommended
        critic_dropout_rate = None, # ISSUES

        generator_batch_norm_momentum = 0.5,
        generator_learning_rate = 0.00005, # around 0.0002 is recommended
        generator_dropout_rate = 0.0001,
        #===================================================================
        # FILTERS, STRIDES, LAYER & KERNEL SIZES

        # critic (IN:128)       (OUT: 64,  32,  16,   8,   4 )
        critic_conv_filters     = [   32,  64, 128, 256, 512 ],
        critic_conv_strides     = [    2,   2,   2,   2,   2 ],
        critic_conv_kernel_size = [    4,   4,   4,   4,   4 ],

        # GENERATOR (IN:4)         (OUT:  8,  16,  32,  64, 128, 128 )
        generator_initial_dense_layer_size = ( 4, 4, 512 ),

        generator_conv_filters     = [  256, 128,  64,  32,  16, C_CH ],
        generator_conv_kernel_size = [    4,   4,   4,   4,   4,    4 ],
        generator_conv_strides     = [    1,   1,   1,   1,   1,    1 ],
        generator_upsample         = [    2,   2,   2,   2,   2,    1 ],# if not '2', uses Conv2DTranspose
        #===================================================================
    )

    x_train = gan.load_dataset( limit = 50 )
    gan.generate_random_image()
    #gan.clear_terminal()
    gan.clean_folders()

    gan.print_model_summaries( critic = True, generator = True )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )

    file_object.write2file( line_list = strings_list_1, open_file = True )
    file_object.write2file( line_list = strings_list_2, close_file = True )

    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 2 )

    session.end_session()

"""
