"""
#===============================================================================
"""
from TF_Environment import TF_Environment
from wgangp_functions import WGANGP
from FileObject import FileObject

C_CH      = 3
IMG_SIZE  = 256
INPUT_DIM = ( IMG_SIZE, IMG_SIZE, C_CH )

filepath = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/data/results/"
filepath += str( IMG_SIZE ) + "/architecture_" + str( IMG_SIZE ) + ".txt"
"""
#===============================================================================
"""

if( __name__ == "__main__" ):

    file_object = FileObject( filepath = filepath )
    session = TF_Environment( gpu_fraction = 0.7 )
    session.run_session()

    gan = WGANGP(
        #===================================================================
        # INITIALIZATION
        gif = True,
        z_dim = 256,
        img_size = IMG_SIZE,
        gif_rate = 100,
        save_rate = 100,
        input_dim = INPUT_DIM,
        optimiser = "rmsprop",#"adam", # { "adam", "rmsprop" }
        batch_size = 10, #36 #20, #10, #10,
        num_generated_imgs = 16,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        #training_dir = "/home/rory/Documents/01_myDatasets/remy/best/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best_200_faces/",
        #===================================================================
        # DISCRIMINATOR (IN:256) (OUT: 256, 128, 128,  64,  32,  16,    8,   4 )

        critic_conv_filters     = [    64,   64,  64,  64,  64,  64 ],
        critic_conv_kernel_size = [      4,   4,   4,   4,   4,   4 ],
        critic_conv_strides     = [      1,   2,   2,   2,   2,   2 ],

        #critic_conv_filters     = [     16,  32,  64, 128, 256,  512 ],
        #critic_conv_kernel_size = [      4,   4,   4,   4,   4,    4 ],
        #critic_conv_strides     = [      1,   2,   2,   2,   2,    2 ],

        #critic_conv_filters     = [     32,  64,  64, 128, 256, 512, 1024 ],
        #critic_conv_kernel_size = [      5,   5,   5,   5,   5,   5,    5 ],
        #critic_conv_strides     = [      1,   2,   1,   2,   2,   2,    2 ],

        critic_batch_norm_momentum = None,
        critic_dropout_rate = None,
        critic_learning_rate = 0.00005, #0.001,
        #===================================================================
        # GENERATOR       (IN: 256) (OUT:   8,   16,  32,  64,  64, 128, 128, 256, 256, 256 )

        #generator_initial_dense_layer_size = ( 8, 8, 512 ),
        generator_initial_dense_layer_size = (  8,  8,  64 ),

        generator_conv_filters     = [     64, 128, 128, 128, 128,  C_CH ],
        generator_conv_kernel_size = [      4,   4,   4,   4,   4,     4 ],
        generator_conv_strides     = [      2,   2,   2,   2,   2,     1 ],
        generator_upsample         = [      1,   1,   1,   1,   1,     1 ],

        #generator_conv_filters     = [    256, 128,  64,  32,  32,  16,  16, C_CH ],
        #generator_conv_kernel_size = [      4,   4,   4,   4,   4,   4,   4,    4 ],
        #generator_conv_strides     = [      2,   2,   2,   2,   1,   2,   1,    1 ],
        #generator_upsample         = [      1,   1,   1,   1,   1,   1,   1,    1 ],

        #generator_conv_filters     = [    256, 128,  64,  64,  32,  32,  16,  16, C_CH ],
        #generator_conv_kernel_size = [      5,   5,   5,   5,   5,   5,   5,   5,    5 ],
        #generator_conv_strides     = [      2,   2,   2,   1,   2,   1,   2,   1,    1 ],
        #generator_upsample         = [      1,   1,   1,   1,   1,   1,   1,   1,    1 ],

        generator_batch_norm_momentum = 0.5,
        generator_dropout_rate = None, #0.01,
        generator_learning_rate = 0.00005 #0.0005
        #===================================================================
    )

    #x_train = gan.load_dataset( dummy_prediction = True )
    x_train = gan.load_dataset( limit = 20 )

    gan.generate_random_image()
    gan.clear_terminal()
    gan.clean_folders()

    #gan.print_model_summaries( critic = True, generator = True )
    strings_list_1 = gan.print_model_layers( architecture = "critic" )
    strings_list_2 = gan.print_model_layers( architecture = "generator" )
    file_object.write2file( line_list = strings_list_1, open_file = True )
    file_object.write2file( line_list = strings_list_2, close_file = True )

    gan.train( x_train = x_train, epochs = 100000, training_ratio = 2 )

    session.end_session()

"""
#===============================================================================
"""
"""
        #===================================================================
        # INITIALIZATION
        new = True,                     # Build new model or load old model
        z_dim = 64,
        img_size = IMG_SIZE,
        save_rate = 100,
        input_dim = INPUT_DIM,
        optimiser = "adam",
        batch_size = 20, #10, #10,
        grad_weight = 10,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best/",
        #===================================================================
        # DISCRIMINATOR # 128 > 64 > 32 >  16 >   8 >   4
        #critic_conv_filters = [ 32 , 64 , 128 , 256 ],
        critic_conv_filters = [  16, 32,  64,  128, 256 ],
        critic_conv_kernel_size = 3,#5,
        critic_conv_strides = 2,
        critic_batch_norm_momentum = None,
        critic_activation = 'leaky_relu',
        critic_dropout_rate = None,
        critic_learning_rate = 0.0005, #0.001,
        #===================================================================
        # GENERATOR
        #generator_initial_dense_layer_size = ( 8, 8, 256 ),
        generator_initial_dense_layer_size = ( 4, 4, 256 ),

        #generator_conv_filters = [ 128, 128, 128, CHANNELS ],
        generator_conv_filters = [ 128, 128, 128, 128, CHANNELS ],
        generator_conv_kernel_size = 3,#5,
        #generator_conv_strides = 2,
        #generator_upsample = 1,
        generator_conv_strides = 1,
        generator_upsample = 2,
        generator_batch_norm_momentum = 0.9, #0.1, #0.9,
        generator_activation = 'leaky_relu',
        generator_dropout_rate = None, #0.01,
        generator_learning_rate = 0.0005 #0.0005
        #===================================================================
"""
"""
        #===================================================================
        Nice Evenly Distributed Noise Generated Images for Epoch_1
        #===================================================================
        # INITIALIZATION
        new = True,                     # Build new model or load old model
        z_dim = 64,
        img_size = IMG_SIZE,
        save_rate = 100,
        input_dim = INPUT_DIM,
        optimiser = "adam",
        batch_size = 36, #10, #10,
        grad_weight = 10,
        working_dir = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        training_dir = "/home/rory/Documents/01_myDatasets/remy/best/",
        #===================================================================
        # DISCRIMINATOR # 128 > 64 > 32 >  16 >   8 >   4
        #critic_conv_filters = [ 32 , 64 , 128 , 256 ],
        critic_conv_filters = [  32,  64,  128 , 256 ],
        critic_conv_kernel_size = 5,
        critic_conv_strides = 2,
        critic_batch_norm_momentum = None,
        critic_activation = 'leaky_relu',
        critic_dropout_rate = None,
        critic_learning_rate = 0.0001, #0.001,
        #===================================================================
        # GENERATOR
        #generator_initial_dense_layer_size = ( 8, 8, 256 ),
        generator_initial_dense_layer_size = ( 8, 8, 256 ),
        #generator_conv_filters = [ 128, 128, 128, CHANNELS ],
        generator_conv_filters = [ 128, 128, 128, CHANNELS ],
        generator_conv_kernel_size = 5,
        generator_conv_strides = 2,
        generator_upsample = 1,
        #generator_conv_strides = 1,
        #generator_upsample = 2,
        generator_batch_norm_momentum = 0.1, #0.9,
        generator_activation = 'leaky_relu',
        generator_dropout_rate = None, #0.01,
        generator_learning_rate = 0.0005 #0.0005
        #===================================================================
"""
