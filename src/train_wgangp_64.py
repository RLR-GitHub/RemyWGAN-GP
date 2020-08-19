"""
#===============================================================================
# WGAN - GP MODEL
#===============================================================================
"""
from TF_Environment import TF_Environment
from FileObject import FileObject
from WGAN_GP import WGAN_GP

C_CH      = 3
IMG_SIZE  = 64
OPTIMIZER = "rmsprop"                               # { "adam", "rmsprop" }
INPUT_DIM = ( IMG_SIZE, IMG_SIZE, C_CH )
filepath  = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/data/results/"
filepath  += str( IMG_SIZE ) + "/architecture_" + str( IMG_SIZE ) + ".txt"
"""
#===============================================================================
"""
if( __name__ == "__main__" ):

    file_object = FileObject( filepath = filepath )
    session = TF_Environment( gpu_fraction = 0.6 )
    session.run_session()

    gan = WGAN_GP(
        #=======================================================================
        z_dim = 100,
        img_size = IMG_SIZE,
        gif_rate = 5000,
        save_rate = 25,
        print_rate = 5,
        input_dim = INPUT_DIM,
        optimizer = OPTIMIZER,
        batch_size = 30,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_300_faces/",
        num_generated_imgs = 16,
        #=======================================================================
        critic_activation = 'leaky_relu',           # { 'relu', 'leaky_relu' }
        critic_dropout_rate = None,                 # ISSUES
        critic_learning_rate = 0.0005,              # { 0.0001, 0.00005 }

        critic_gaussian_noise = False,              # CHANGE BACK TO TRUE !!!!!

        generator_activation = 'relu',              # { 'relu', 'leaky_relu' }
        generator_dropout_rate = None,
        generator_learning_rate = 0.0005,           # { 0.0001, 0.00005 }

        generator_instance_norm = True,
        generator_layer_normalization = False,      # CHANGE BACK TO TRUE
        generator_batch_norm_momentum = None,       # REQUIRES LARGE DATASET
        #=======================================================================

        #critic_conv_filters     = [  64, 128, 256 ],

        critic_conv_filters     = [  32,  64, 128 ],
        critic_conv_strides     = [   2,   2,   2 ],
        critic_conv_kernel_size = [   7,   5,   3 ],

        #generator_conv_filters     = [  128, 128, 128, C_CH ],

        generator_initial_dense_layer_size = (  8,  8,  64 ),
        generator_conv_filters     = [   64,  64,  64, C_CH ],
        generator_conv_kernel_size = [    4,   4,   4,   4  ],
        generator_conv_strides     = [    2,   2,   2,   1  ],
        generator_upsample         = [    1,   1,   1,   1  ],
        generator_conv_type        = [   't', 't', 't', 't' ],  # { 'c', 't' }

        #=======================================================================
    )

    x_train = gan.load_dataset( limit = 600, augment = True )

    gan.generate_random_image()
    gan.clear_terminal()
    gan.clean_folders()

    strings_list_0 = gan.print_model_layers( architecture = "critic" )
    strings_list_1 = gan.print_model_layers( architecture = "generator" )
    strings_list_2 = gan.print_model_layers( architecture = "parameters" )

    #gan.print_model_summaries( critic = True, generator = True )

    file_object.write2file( line_list = strings_list_0, open_file =  True, close_file = False )
    file_object.write2file( line_list = strings_list_1, open_file = False, close_file = False )
    file_object.write2file( line_list = strings_list_2, open_file = False, close_file =  True )

    #gan.graph_update_rates( epoch_arr = [ 100, 250, 1000, 10000 ], update_rate = [ 50, 100, 200, 250 ] )
    gan.graph_update_rates( epoch_arr = [ 100, 1000, 10000 ], update_rate = [ 250, 500, 1000 ] )

    if( OPTIMIZER == "adam" ): gan.lr_reductions( epoch_arr = [ 1000 ], lr_arr = [ 0.0005 ] ) # NO NEED TO CHANGE LR FOR RMSPROP

    gan.train( x_train = x_train, epochs = 1000000, training_ratio = 1 )
    session.end_session()

"""
#===============================================================================
# OLD JUNK PARAMETERS
#===============================================================================
"""


"""
        #critic_conv_filters     = [  64, 128, 256, 512 ],
        #critic_conv_strides     = [   2,   2,   2,   1 ],
        #critic_conv_kernel_size = [   4,   4,   4,   4 ],

        #generator_initial_dense_layer_size = (  4,  4, 128 ),
        #generator_initial_dense_layer_size = (  8,  8,  64 ),

        #generator_conv_filters     = [  128, 128, 128, 128, C_CH ],
        #generator_conv_kernel_size = [    4,   4,   4,   4,   4  ],
        #generator_conv_strides     = [    1,   1,   1,   1,   1  ],
        #generator_upsample         = [    2,   2,   2,   2,   1  ],
        #generator_conv_type        = [   'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [   64,  64,  64,  64,  64,  64,  C_CH ],
        #generator_conv_kernel_size = [    3,   3,   3,   3,   3,   3,    3  ],
        #generator_conv_strides     = [    2,   1,   2,   1,   2,   1,    1  ],
        #generator_upsample         = [    1,   1,   1,   1,   1,   1,    1  ],
        #generator_conv_type        = [   't', 't', 't', 't', 't', 't',  't' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [  32,  32,  32,  32,  32,  32,  32,  32,  32, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   1,   2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed

        #generator_conv_filters     = [ 512, 512, 256, 256, 128, 128,  64,  64,  32, C_CH ],
        #generator_conv_kernel_size = [   3,   3,   3,   3,   3,   3,   3,   3,   5,   7  ],
        #generator_conv_strides     = [   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],
        #generator_upsample         = [   1,   2,   1,   2,   1,   2,   1,   2,   1,   1  ],
        #generator_conv_type        = [  'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c' ], # 'c' for regular, 't' for transposed
"""

"""
#===============================================================================
#===============================================================================
"""
