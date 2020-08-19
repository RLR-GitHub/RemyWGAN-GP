import os
import time
import glob
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
from GIF_Creator import GIF_Creator
from scipy import stats


from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.initializers import RandomNormal
import tensorflow_addons as tfa #https://www.tensorflow.org/addons/tutorials/layers_normalizations

header_1 = "\n================================================================="
header_2 = "\n-----------------------------------------------------------------"

class RandomWeightedAverage( _Merge ):
    def __init__( self, batch_size ):
        super().__init__()
        self.batch_size = batch_size

    def _merge_function( self, inputs ):
        weights = tf.keras.backend.random_uniform( ( self.batch_size, 1, 1, 1 ) )
        return( ( weights * inputs[ 0 ] ) + ( ( 1 - weights ) * inputs[ 1 ] ) )

class WGAN_GP():
    def __init__(
        self,
        new : bool = True,# Build new model or load old model
        gif : bool = True,
        z_dim : int = 100,
        img_size : int = 64,
        color_chs : int = 3,
        gif_rate : int = 1000,
        save_rate : int = 100,
        print_rate : int = 5,
        input_dim : tuple = ( 64, 64, 3 ),
        optimizer : str = "adam",
        batch_size : int = 32,
        grad_weight : int = 10,
        working_dir : str = "/home/rory/Documents/00_myPrograms/RemyWGAN-GP/",
        training_dir : str = "/home/rory/Documents/01_myDatasets/remy/best_faces_augmented/",
        num_generated_imgs : int = 16,

        critic_conv_filters : list = [ 32, 64, 128 ],
        critic_conv_strides : int = 2,
        critic_conv_kernel_size : int = 4,
        critic_batch_norm_momentum : float = None, # NEVER USE FOR WGAN-GP
        critic_activation : str = "leaky_relu",
        critic_dropout_rate : float = None,
        critic_learning_rate : float = 0.0001,
        critic_gaussian_noise : bool = False,

        generator_initial_dense_layer_size : tuple = ( 8, 8, 128 ),
        generator_conv_filters : list = [ 64, 32, 3 ],
        generator_conv_strides : int = 2,
        generator_conv_kernel_size : int = 1,
        generator_upsample : int = 1, # if not '2', uses Conv2DTranspose
        generator_conv_type : list = [  'c', 'c', 'c', 'c',   'c' ], # 'c' for regular, 't' for transposed

        generator_instance_norm : bool = False,
        generator_layer_normalization : bool = False,
        generator_batch_norm_momentum : float = None,
        generator_activation : str = "leaky_relu",
        generator_dropout_rate : float = None,
        generator_learning_rate : float = 0.0001
    ):

        self.new = new
        self.gif = gif
        self.name = 'gan'
        self.z_dim = z_dim
        self.img_size = img_size
        self.gif_rate = gif_rate
        self.color_chs = color_chs
        self.save_rate = save_rate
        self.print_rate = print_rate
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.grad_weight = grad_weight
        self.working_dir = working_dir
        self.training_dir = training_dir
        self.num_generated_imgs = num_generated_imgs

        self.critic_conv_filters = critic_conv_filters

        self.critic_conv_kernel_size = self._set_param_array( critic_conv_kernel_size, critic_conv_filters )
        self.critic_conv_strides = self._set_param_array( critic_conv_strides, critic_conv_filters )

        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate
        self.critic_gaussian_noise = critic_gaussian_noise

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_conv_filters = generator_conv_filters

        self.generator_conv_kernel_size = self._set_param_array( generator_conv_kernel_size, generator_conv_filters )
        self.generator_conv_strides = self._set_param_array( generator_conv_strides, generator_conv_filters )
        self.generator_upsample = self._set_param_array( generator_upsample, generator_conv_filters )
        self.generator_conv_type = self._set_param_array( generator_conv_type, generator_conv_filters )

        self.generator_instance_norm = generator_instance_norm
        self.generator_layer_normalization = generator_layer_normalization
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.n_layers_critic = len(critic_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = tf.keras.initializers.RandomNormal( mean = 0.0, stddev = 0.02 ) #mean = 0.0, stddev = 0.1 )
        #self.weight_init = tf.keras.initializers.RandomNormal( mean = 0.0, stddev = 0.05 ) #mean = 0.0, stddev = 0.1 )

        self.fig_path = self.working_dir + "data/results/" + str( self.img_size  ) + "/" + "graph_performance_" + str( self.img_size  ) + ".png"
        self.models_dir = self.working_dir + "data/models/" + str( self.img_size  ) + "/"
        self.pickles_dir = self.working_dir + "data/pickles/" + str( self.img_size  ) + "/"

        self.results_dir = self.working_dir + "data/results/" + str( self.img_size  ) + "/"
        self.test_image_path = self.working_dir + "data/results/" + str( self.img_size  ) + "/" + "dataset_sample.png"

        self.gif_result_path_single = self.results_dir + 'animation_single_' + str( self.img_size  ) + '.gif'
        self.gif_result_path_batch = self.results_dir + 'animation_batch_' + str( self.img_size  ) + '.gif'

        self.gif_images_path_single = self.results_dir + 'gif_images/single/'
        self.gif_images_path_batch = self.results_dir + 'gif_images/batch/'

        self.gif_creator_single = GIF_Creator( filepath = self.gif_result_path_single, imgspath = self.gif_images_path_single )
        self.gif_creator_batch = GIF_Creator( filepath = self.gif_result_path_batch, imgspath = self.gif_images_path_batch )

        self.graph_update_rate = None
        self.lr_epoch_arr  = None
        self.g_losses = []
        self.d_losses = []
        self.epoch = 0

        self._build_adversarial()
        self._set_training_label_vectors()

    def gradient_penalty_loss( self, y_true, y_pred, averaged_samples, gradient_penalty_weight ):

        gradients = tf.keras.backend.gradients( y_pred, averaged_samples )[ 0 ]
        gradients_squared = tf.keras.backend.square( gradients )    # Compute euclidean norm by squaring
        gradients_squared_sum = tf.keras.backend.sum( gradients_squared, axis = np.arange( 1, len( gradients_squared.shape ) ) )   # Summing them over the rows
        gradients_l2_norm = tf.keras.backend.sqrt( gradients_squared_sum ) # Sqaure root
        #gradient_penalty = gradient_penalty_weight * tf.keras.backend.square( 1 - gradients_l2_norm ) # lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.keras.backend.square( 1 - gradients_l2_norm ) # lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty_mean_loss = tf.keras.backend.mean( gradient_penalty ) # Return mean loss over all batch samples
        return( gradient_penalty_mean_loss )

    def wasserstein( self, y_true, y_pred ):
        wasserstein_loss = tf.keras.backend.mean( y_true * y_pred )
        #wasserstein_loss = -1.0 * wasserstein_loss # Flip values ??? CHANGE_LATER MAYBEEEE??????
        wasserstein_loss = wasserstein_loss # Flip values ??? CHANGE_LATER MAYBEEEE??????
        return( wasserstein_loss )

    def _set_param_array( self, object, layers ):
        if( type( object ) is int ): arr = [ object for i in range( 0, len( layers ) + 1 ) ]
        elif( type( object ) is str ): arr = [ object for i in range( 0, len( layers ) + 1 ) ]
        else: arr = object
        return( arr )

    """
    #===========================================================================
    # STANDARD MODEL LAYERS
    #===========================================================================
    """

    """
    #x = tf.keras.layers.BatchNormalization( axis = bn_axis, momentum = self.generator_batch_norm_momentum )( x )
    #x = tf.keras.layers.BatchNormalization( axis = bn_axis, momentum = self.generator_batch_norm_momentum, trainable = True )( x )
    #x = tf.keras.layers.BatchNormalization( momentum = self.generator_batch_norm_momentum, axis = bn_axis )( x )
    #x = tf.keras.layers.BatchNormalization( momentum = self.generator_batch_norm_momentum )( x, training = True )
    """

    # Pixelwise feature vector normalization.
    #def pixel_norm( self, inputs, epsilon = 1e-8 ):
        #square = inputs ** 2
        #mean = tf.keras.backend.mean( square, axis = -1, keepdims = True )
        #normalization_constant = tf.keras.backend.sqrt( mean + epsilon )
        #return inputs / normalization_constant

    def model_input_layer( self, shape, name ): return( tf.keras.layers.Input( shape = shape, name = name ) )

    def reshape_layer( self, input, shape ):

        keras_channel_order = tf.keras.backend.image_data_format() # { 'channels_first', 'channels_last' }
        #print( "Required Channel Order (for generator input): ", keras_channel_order )

        if( keras_channel_order == 'channels_first' ):
            output = tf.keras.layers.Reshape( ( shape[ 2 ], shape[ 1 ], shape[ 0 ] ) )( input )
            bn_axis = 1
        else:
            output = tf.keras.layers.Reshape( shape )( input )
            bn_axis = -1 # Set axis to be normalized (usually the features axis )
        return( output )

    def activation_layer( self, input, activation ):
        if activation == 'leaky_relu': output = tf.keras.layers.LeakyReLU( alpha = 0.2 )( input )
        else: output = tf.keras.layers.Activation( activation )( input )
        return( output )

    # GAUSSIAN_NOISE -- ADD TO INPUT OF DISCRIMINATOR TO PREVENT OVERFITTING
    def gaussian_noise_layer( self, input, std = 0.1 ):
        noise = tf.keras.backend.random_normal( shape = tf.shape( input ), mean = 0.0, stddev = std, dtype = tf.float32 )
        return( input + noise )

    def upsample_layer(self, input): return( tf.keras.layers.UpSampling2D( interpolation = 'nearest' )( input ) ) #tf.keras.layers.UpSampling2D()( x )

    def batch_normalization_layer(self, input, momentum): return( tf.keras.layers.BatchNormalization( momentum = momentum )( input ) )
    def layer_normalization_layer(self, input): return( tf.keras.layers.LayerNormalization( axis = -1, epsilon = 1e-8, center = True, scale = True )( input ) )
    def instance_normalization_layer(self, input): return( tfa.layers.InstanceNormalization( axis = -1, center = False, scale = False )( input ) )

    def dense_layer( self, input, units, activation = None ): return( tf.keras.layers.Dense( units = units, activation = activation, kernel_initializer = self.weight_init )( input ) )
    def flatten_layer( self, input ): return( tf.keras.layers.Flatten()( input ) )
    def dropout_layer(self, input, rate): return( tf.keras.layers.Dropout( rate = rate )( input ) )

    def build_model( self, input, output, name ): return( tf.keras.models.Model( input, output, name = name ) )

    """
    #===========================================================================
    BUILD THE critic -- CRITIC -- Batch norm shouldn't be used in WGAN-GP critic
    #===========================================================================
    """

    def _build_critic( self ):

        use_gaussnoise = self.critic_gaussian_noise

        def DiscConv( input ): return(
            tf.keras.layers.Conv2D(
                filters = self.critic_conv_filters[ i ],
                kernel_size = self.critic_conv_kernel_size[ i ],
                strides = self.critic_conv_strides[ i ],
                padding = 'same',
                name = 'critic_conv_' + str( i ),
                kernel_initializer = self.weight_init
            )( input )
        )

        #===========================================================================
        image_input = self.model_input_layer( shape = self.input_dim, name = 'critic_input' )
        x = image_input

        x = self.gaussian_noise_layer( input = x, std = 0.1 ) if( use_gaussnoise ) else( x )
        #x = self.gaussian_noise_layer( input = x, std = 0.2 ) if( use_gaussnoise ) else( x )

        for i in range( self.n_layers_critic ):

            x = DiscConv( x )
            #x = self.instance_normalization_layer( input = x ) if( use_instancenorm ) else( x )
            x = self.activation_layer( input = x, activation = self.critic_activation )
            #x = self.gaussian_noise_layer( input = x, std = 0.05 ) # MAYBE TAKE OUT
            #droput

        x = self.flatten_layer( input = x )

        #x = self.gaussian_noise_layer( input = x, std = 0.05 )# MAYBE TAKE OUT
        #x = self.dense_layer( input = x, units = self.critic_conv_filters[ -1 ], activation = None )# MAYBE TAKE OUT
        #x = self.activation_layer( input = x, activation = self.critic_activation )# MAYBE TAKE OUT

        validity_output = self.dense_layer( input = x, units = 1, activation = None )

        self.critic = self.build_model( input = image_input, output = validity_output, name = "critic" )
        assert self.critic.input_shape == ( None, self.img_size, self.img_size, 3 )

    """
    #===========================================================================
    BUILD THE GENERATOR
    #===========================================================================
    """

    def _build_generator( self ):

        use_dropout = self.generator_dropout_rate
        use_instancenorm = self.generator_instance_norm
        use_batchnorm = self.generator_batch_norm_momentum
        use_layernorm = self.generator_layer_normalization

        def GenConv(input): return(
            tf.keras.layers.Conv2D(
                filters = self.generator_conv_filters[ i ],
                kernel_size = self.generator_conv_kernel_size[ i ],
                padding = 'same',
                strides = self.generator_conv_strides[i],
                name = 'generator_conv_' + str( i ),
                kernel_initializer = self.weight_init
            )( input )
        )

        def GenTransposedConv(input): return(
            tf.keras.layers.Conv2DTranspose(
                filters = self.generator_conv_filters[ i ],
                kernel_size = self.generator_conv_kernel_size[ i ],
                padding = 'same',
                strides = self.generator_conv_strides[i],
                name = 'generator_t_conv_' + str( i ),
                kernel_initializer = self.weight_init
            )( input )
        )

        convolution = lambda input : GenTransposedConv( input ) if( self.generator_conv_type[ i ] == 't' ) else( GenConv( input ) )

        #=======================================================================

        noise_input = self.model_input_layer( shape = ( self.z_dim, ), name = 'generator_input' ) # noise
        x = noise_input

        x = self.dense_layer( input = x, units = np.prod( self.generator_initial_dense_layer_size ), activation = None )

        #x = self.batch_normalization_layer( input = x, momentum = self.generator_batch_norm_momentum ) if( use_batchnorm ) else( x )
        #x = self.layer_normalization_layer( input = x ) if( use_layernorm ) else( x )
        #x = self.instance_normalization_layer( input = x ) if( use_instancenorm ) else( x )

        x = self.activation_layer( input = x, activation = self.generator_activation )


        x = self.reshape_layer( input = x, shape = self.generator_initial_dense_layer_size )
        #x = self.dropout_layer( input = x, rate = self.generator_dropout_rate ) if( use_dropout ) else( x )

        for i in range( self.n_layers_generator ):

            x = self.upsample_layer( input = x ) if( self.generator_upsample[ i ] == 2 ) else( x )
            x = convolution( input = x )

            if( i < int( self.n_layers_generator - 1 ) ):
                #x = self.batch_normalization_layer( input = x, momentum = self.generator_batch_norm_momentum ) if( use_batchnorm ) else( x )
                #x = self.layer_normalization_layer( input = x ) if( use_layernorm ) else( x )
                x = self.instance_normalization_layer( input = x ) if( use_instancenorm ) else( x )

                x = self.activation_layer( input = x, activation = self.generator_activation )
                #x = self.activation_layer( input = x, activation = self.generator_activation )
                #x = self.layer_normalization_layer( input = x ) if( use_layernorm ) else( x )
            else:
                x = self.activation_layer( input = x, activation = 'tanh' )

        image_output = x
        self.generator = self.build_model( input = noise_input, output = image_output, name = "generator" )
        assert self.generator.output_shape == ( None, self.img_size, self.img_size, 3 )

    """
    #===========================================================================
    #===========================================================================
    """

    def get_optimizer( self, lr : float, beta_1 : float = 0.5 ):#0.75 ) # default is 0.9 !!!!!! CHANGE_LATER

        if self.optimizer == 'adam': optimizer = tf.keras.optimizers.Adam( lr = lr, beta_1 = 0.5, beta_2 = 0.9 ) #( lr = lr, beta_1 = beta_1 ) CHANGE BACK ???!!!!!!!
        elif self.optimizer == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop( lr = lr )
        #else: optimizer = tf.keras.optimizers.Adam( lr = lr )
        return( optimizer )

    def set_trainable_layers( self, model, train ):

        for layer in model.layers:
            layer.trainable = train

        model.trainable = train
    """
    #===========================================================================
    BUILD THE GAN MODEL
    #===========================================================================
    """
    def _build_adversarial( self ):

        # INITIALIZE GENERATOR & critic
        self._build_generator()
        self._build_critic()

        #-----------------------------------------------------------------------
        # CONSTRUCT CRITIC MODEL
        #-----------------------------------------------------------------------

        self.set_trainable_layers( model = self.generator, train = False )

        real_samples = tf.keras.layers.Input( shape = self.input_dim ) # Image input (real sample)
        generator_input_for_critic = tf.keras.layers.Input( shape = ( self.z_dim, ) ) # Noise input

        generator_samples_for_critic = self.generator( generator_input_for_critic ) # Generated image from noise (fake sample)

        # Pass in real & fake images, calculate Wasserstein loss
        critic_output_from_real_samples = self.critic( real_samples )
        critic_output_from_generator = self.critic( generator_samples_for_critic )

        # Create interpolated image ( weighted average btwn real & fake images )
        averaged_samples = RandomWeightedAverage( batch_size = self.batch_size )( [ real_samples, generator_samples_for_critic ] ) # Interpolated image
        averaged_samples_out = self.critic( averaged_samples ) # Determine validity of weighted sample

        # Use Python partial to pass interpolated images through the gradient penalty loss function
        partial_gp_loss = partial( self.gradient_penalty_loss, averaged_samples = averaged_samples, gradient_penalty_weight = self.grad_weight )
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function name

        self.critic_model = tf.keras.models.Model(
            inputs = [ real_samples, generator_input_for_critic ],
            outputs = [ critic_output_from_real_samples, critic_output_from_generator, averaged_samples_out ]
        )

        self.critic_model.compile(
            loss = [ self.wasserstein, self.wasserstein, partial_gp_loss ],
            optimizer = self.get_optimizer( lr = self.critic_learning_rate ),
            loss_weights= [ 1, 1, 10 ]
        )

        #-----------------------------------------------------------------------
        # CONSTRUCT GENERATOR MODEL
        #-----------------------------------------------------------------------
        # generator_model is used to train generator layers...
        # --> thus critic layers must not be trainable
        # --> after model is compiled, changing .trainable will have no effect
        # --> as such, setting critic.trainable = True won't be an issue
        # --> as long as the generator_model is compiled first
        #-----------------------------------------------------------------------

        self.set_trainable_layers( model = self.critic, train = False )
        self.set_trainable_layers( model = self.generator, train = True )

        generator_input = tf.keras.layers.Input( shape = ( self.z_dim, ) ) # Fake image sample
        generator_layers = self.generator( generator_input )               # Returns fake img

        critic_validity_score = self.critic( generator_layers )       # Get validity of generated image

        self.generator_model = tf.keras.models.Model(
            inputs = generator_input,
            outputs = critic_validity_score,
        )

        self.generator_model.compile(
            loss = self.wasserstein,
            optimizer = self.get_optimizer( lr = self.generator_learning_rate )
        )


    def _set_training_label_vectors( self ): # Adversarial ground truth labels
        self.real_label = -1.0 * np.ones(  ( self.batch_size, 1 ), dtype = np.float32 )
        self.fake_label =  1.0 * np.ones(  ( self.batch_size, 1 ), dtype = np.float32 )
        self.dumb_label =  1.0 * np.zeros( ( self.batch_size, 1 ), dtype = np.float32 ) # Dummy gt for gp

    """
    #===========================================================================
    TRAINING THE GENERATOR
    #===========================================================================
    """
    def random_data_batch( self, data ):
        #image_batch = next(x_train)[0]
        #if image_batch.shape[0] != batch_size: image_batch = next(x_train)[0]

        # Select random batch of images
        idx = np.random.randint( low = 0, high = data.shape[ 0 ], size = self.batch_size )
        image_batch = data[idx]
        #print( "indices: ", idx )

        return( image_batch )
    """
    #===========================================================================
    TRAINING THE GENERATOR
    #===========================================================================
    """
    def train_generator( self, input_noise ):
        g_loss = self.generator_model.train_on_batch( input_noise, self.real_label )
        self.g_losses.append( g_loss )
    """
    #===========================================================================
    TRAINING THE critic -- CRITIC
    #===========================================================================
    """

    def train_critic( self, input_noise, input_images ):
        d_loss = self.critic_model.train_on_batch(
            [ input_images, input_noise ],
            [ self.real_label, self.fake_label, self.dumb_label ]
        )
        self.d_losses.append( d_loss )
    """
    #===========================================================================
    TRAINING THE WGAN MODEL
    #===========================================================================
    """
    def graph_update_rates( self, epoch_arr : list = [ 100, 5000, 10000 ], update_rate : list = [ 0.0005, 0.0005, 0.0005 ] ):
        self.graph_update_rate = update_rate
        self.graph_epoch_arr = epoch_arr

    def lr_reductions( self, epoch_arr : list = [ 100, 5000, 10000 ], lr_arr : list = [ 0.0005, 0.0005, 0.0005 ] ):
        print( "Learning Rate Reudctions: ", lr_arr )
        self.new_lr_arr = lr_arr
        self.lr_epoch_arr = epoch_arr

    def train( self, x_train, epochs, training_ratio = 5 ):

        time_change = lambda previous : time.time() - previous

        loss_str = "Epoch_%s [D: %s (r: %s, f: %s)] [G: %s]\t[GP: %s]"
        time_str = "\t[Time: %s (t_disc: %s, t_gen: %s)]"
        header = "\nTRAINING WGAN-GP MODEL"

        opening_header_block = header_1 + header + header_2 + "\n"
        closing_header_block = header_1 + "\n"

        gnd_truth_img = x_train[ 0 ]
        dataset_size = int( x_train.shape[ 0 ] )
        minibatches_size = int( self.batch_size * training_ratio )
        batches_per_epoch = int( dataset_size // minibatches_size )

        print( opening_header_block )
        print( "Minibatches Per Epoch (Critic D) : {}".format( minibatches_size ) )
        print( "Batches Per Epoch (Generator G): {}".format( batches_per_epoch ) )
        print( closing_header_block )

        for epoch in range( self.epoch, self.epoch + epochs ):

            start_time = time.time()
            pos_prediction_msg = ""
            #np.random.shuffle( x_train )

            """
            #===================================================================
            TRAINING LOOP -- RANDOM BATCH WITHIN DATASET
            #===================================================================
            """

            #"""
            for _ in range( training_ratio ):

                noise = np.random.normal( 0, 1, ( self.batch_size, self.z_dim ) ) # Random sample vector from latent space
                random_image_batch = self.random_data_batch( data = x_train )

                self.train_critic( input_noise = noise, input_images = random_image_batch )

            c_time = time_change( previous = start_time )
            self.train_generator( input_noise = noise )
            g_time = time_change( previous = ( c_time + start_time ) )

            if( epoch == 0 ): self.save_image_sample( random_image_batch )

            #"""

            """
            #===================================================================
            TRAINING LOOP -- ENTIRE DATASET
            #===================================================================
            """
            """
            np.random.shuffle( x_train )

            # Loop through all image batches of dataset
            for i in range( batches_per_epoch ):

                discriminator_minibatches = x_train[ i * minibatches_size : ( i + 1 ) * minibatches_size ]

                for j in range( training_ratio ):
                    minibatch = discriminator_minibatches[ j * self.batch_size : ( j + 1 ) * self.batch_size ]

                    noise = np.random.normal( 0, 1, ( self.batch_size, self.z_dim ) ) # Random sample vector from latent space
                    self.train_critic( input_noise = noise, input_images = minibatch )

                c_time = time_change( previous = start_time )
                self.train_generator( input_noise = noise )
                g_time = time_change( previous = ( c_time + start_time ) )

            if( epoch == 0 ): self.save_image_sample( x_train )
            """
            """
            #===================================================================
            """

            if( epoch % self.save_rate == 0 ):
                self.graph_performance( epoch, batches_per_epoch, training_ratio )
                pos_prediction_msg = self.generate_and_save_images( epoch, gnd_truth_img = gnd_truth_img )

            if( self.lr_epoch_arr != None ):
                for k in range( len( self.lr_epoch_arr ) ):
                    if( epoch == self.lr_epoch_arr[ k ] ): self.learning_rate_change( new_lr = self.new_lr_arr[ k ] )

            if( self.graph_update_rate != None ):
                for k in range( len( self.graph_update_rate ) ):
                    if( epoch == self.graph_epoch_arr[ k ] ): self.save_rate = self.graph_update_rate[ k ]

            d_loss = self.d_losses[ -1 ]
            g_loss = self.g_losses[ -1 ]
            t_diff = time_change( previous = start_time )

            if( epoch % self.print_rate == 0 ):
                print(
                    loss_str % (
                        self.num2str( string = self.epoch, round_val = 1, just_val = 7, just_dir = 'l' ),
                        self.num2str( string =  d_loss[0], round_val = 1, just_val = 7, just_dir = 'r' ),
                        self.num2str( string =  d_loss[1], round_val = 1, just_val = 7, just_dir = 'r' ),
                        self.num2str( string =  d_loss[2], round_val = 1, just_val = 7, just_dir = 'r' ),
                        self.num2str( string =     g_loss, round_val = 1, just_val = 8, just_dir = 'r' ),
                        self.num2str( string =  d_loss[3], round_val = 1, just_val = 6, just_dir = 'r' )
                    ),
                    time_str % (
                        self.num2str( string = t_diff, round_val = 1, just_val = 4, just_dir = 'r' ),
                        self.num2str( string = c_time, round_val = 2, just_val = 4, just_dir = 'r' ),
                        self.num2str( string = g_time, round_val = 2, just_val = 4, just_dir = 'r' )
                    ),
                    pos_prediction_msg
                )

            self.epoch += 1
        self.generate_and_save_images()
    """
    #===========================================================================
    OTHER
    #===========================================================================
    """
    def learning_rate_change( self, new_lr ):

        print( "\nLEARNING_RATE_REDUCTION")

        print( "\nDiscriminator (OLD):", tf.keras.backend.eval( self.critic_model.optimizer.lr ) )
        print( "Generator (OLD):", tf.keras.backend.eval( self.generator_model.optimizer.lr ) )

        tf.keras.backend.set_value(self.generator_model.optimizer.learning_rate, new_lr)
        tf.keras.backend.set_value(self.critic_model.optimizer.learning_rate, new_lr)

        print( "\nDiscriminator (NEW):", tf.keras.backend.eval( self.critic_model.optimizer.lr ) )
        print( "Generator (NEW):", tf.keras.backend.eval( self.generator_model.optimizer.lr ), "\n" )

    def save_image_sample( self, image_set ) -> None: #CHANGE_LATER

        if( len( image_set ) > 10 ): n = 10
        else: n = len( image_set )

        fig, axes = plt.subplots( figsize = (20,10), ncols = n, nrows = 1 )

        for i in range( n ):
            img = image_set[ i ]
            img = self.data2uint8( img )

            axes[ i ].grid( True )
            axes[ i ].axis( 'off' )
            axes[ i ].imshow( img, interpolation = 'nearest' ) #CHANGE_LATER

        fig.savefig( self.test_image_path, bbox_inches = 'tight' )

    def num2str( self, string, round_val = 3, just_val = 5, just_dir = 'r' ) -> str:
        if( just_dir == 'r' ): string_out = str( round( string, round_val ) ).rjust( just_val )
        else: string_out = str( round( string, round_val ) ).ljust( just_val )
        return( string_out )

    def generate_and_save_images( self, epoch = None, gnd_truth_img = None ):

        fig = plt.figure( figsize = ( 16, 16 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )

        recent = self.results_dir + 'recent.png'
        #passing_fakes = self.results_dir + 'passing_fakes/'
        #all_images = self.results_dir + 'all/' + 'epoch_{:04d}.png'.format( self.epoch )
        gif_image_batch_path = self.gif_images_path_batch + 'image_{:07d}.png'.format( self.epoch )
        gif_image_single_path = self.gif_images_path_single + 'image_{:07d}.png'.format( self.epoch )

        np.random.seed(1)
        sample_noise = np.random.normal( 0, 1, ( self.num_generated_imgs, self.z_dim ) )

        generated_images = self.generator.predict( sample_noise )
        decisions = self.critic.predict( generated_images )

        prediction_msg = ""
        decision_arr, index_arr = list(), list()
        #decision_arr = []
        #index_arr = []
        pos_cnt = 0

        # Rescale images from 0 - 1
        gen_imgs = 0.5 * (generated_images + 1) # From [-1,1] to [0,1]
        gen_imgs = np.clip(gen_imgs, 0, 1)      # Make sure vals are between 0,1

        for i in range( len( generated_images ) ):

            label = decisions[ i ]
            ax = fig.add_subplot( 4, 4, i + 1 )
            img = np.squeeze( gen_imgs[ i, :, :, : ] ) # Squeeze down to 3 dim

            ax.imshow( img, cmap = None, interpolation = 'nearest', vmin = 0, vmax = 1 )
            ax.set_title( label )
            ax.axis( 'off' )

            if( label > 0 ): pos_cnt += 1

            index_arr.append( i )
            decision_arr.append( label )

        fig.savefig( recent, bbox_inches = 'tight', quality = 90 ) #linewidth = 15, edgecolor = 'black',
        fig.savefig( gif_image_batch_path, bbox_inches = 'tight', quality = 100 ) #linewidth = 15, edgecolor = 'black',
        plt.close()


        decision_arr_index = np.argmax( decision_arr )
        generated_img_index = index_arr[ decision_arr_index ]
        val = max( decision_arr )[ 0 ]

        prediction_msg = "[Critic_Prediction: %s]" % ( self.num2str( string = val, round_val = 2, just_val = 7, just_dir = 'r' ) )
        prediction_msg += "[Pos_Cnt: %s]" % ( self.num2str( string = pos_cnt, round_val = 0, just_val = 2, just_dir = 'r' ) )

        img = self.data2uint8( generated_images[ 0 ] ) # Always save img from same index

        fig = plt.figure( figsize = ( 4, 4 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )
        ax = fig.add_subplot( 1, 1, 1 )
        ax.imshow( img, cmap = None, interpolation = 'nearest' )
        ax.axis('off')
        fig.savefig(gif_image_single_path, bbox_inches = 'tight', quality = 100)#linewidth = 15, edgecolor = 'black',
        plt.close()

        if( self.gif == True and epoch % self.gif_rate == 0 ):
            self.gif_creator_batch.set()
            self.gif_creator_single.set()


        gt_img = np.expand_dims( gnd_truth_img, axis = 0 )
        gnd_truth_decision = self.critic.predict( gt_img )[ 0 ][ 0 ]
        prediction_msg += "[GT: %s]" % ( self.num2str( string = gnd_truth_decision, round_val = 2, just_val = 7, just_dir = 'r' ) )

        return( prediction_msg )

    def graph_performance( self, epoch, bpe, training_ratio ):

        figsize = ( 28, 14 )

        #num_batches = int( ( epoch + 1 ) * bpe )                # REGULAR BATCH COUNT, starting from zero
        #num_minibatches = int( training_ratio * num_batches )   # MINIBATCHES (disc)

        #x_gen = np.linspace( start = 0, stop = num_batches, num = num_batches )
        #x_disc = np.linspace( start = 0, stop = num_batches, num = num_minibatches )

        fig, ( ax1, ax2 ) = plt.subplots( figsize = figsize, ncols = 1, nrows = 2, facecolor = 'white' )

        y0 = [ x[ 0 ] for x in self.d_losses ]
        y1 = [ x[ 1 ] for x in self.d_losses ]
        y2 = [ x[ 2 ] for x in self.d_losses ]
        y3 = self.g_losses

        ratio = int( len( self.d_losses ) // len( self.g_losses ) )

        num_batches = int( len( y3 ) )                # REGULAR BATCH COUNT, starting from zero
        num_minibatches = int( training_ratio * num_batches )   # MINIBATCHES (disc)

        x_gen = np.linspace( start = 0, stop = num_batches, num = num_batches )
        x_disc = np.linspace( start = 0, stop = num_batches, num = num_minibatches )

        ax1.plot( x_disc, y0, label='critic Loss Avgerage', color = 'gray', linewidth = 0.25 )
        ax1.plot( x_disc, y1, label='critic Loss (Real)', color = 'lime', linewidth = 0.25 ) # Green for correct / real
        ax1.plot( x_disc, y2, label='critic Loss (Fake)', color = 'red', linewidth = 0.25 ) # Red for fake
        ax1.plot(  x_gen, y3, label='Generator Loss', color = 'yellow', linewidth = 0.25 )

        ax1.axhline( y = 0, linestyle = '--', color = 'black', linewidth = 0.5 ) # add horizontal line
        #ax.axvline( x, linestyle = '--', color = 'k' ) # add vertical line

        ax1.set_title( 'Generator vs Critic (Discriminator)', fontsize = 20 )
        ax1.set_xlabel('Batch Count (bpe: %s)' % bpe, fontsize = 14)
        ax1.set_ylabel('Wasserstein Loss', fontsize = 14)
        ax1.legend(loc='bottom right', fontsize = 'large')

        #=======================================================================

        clip = lambda arr, min, max : np.clip( arr, a_min = min, a_max = max )

        avg = lambda arr : arr.mean()
        std = lambda arr : abs( np.std( arr, axis = None ) )

        sigma = lambda arr, n = 1 : n * abs( std( arr ) )

        #sigma = lambda arr, n = 3 : n * abs( abs( std( arr ) ) - abs( avg( arr ) ) )
        limit = lambda arr, n = 1 : avg( arr ) + sigma( arr, n ) if( avg( arr ) > 0 ) else( avg( arr ) - sigma( arr, n ) )

        #=======================================================================

        #y_val_arr = np.array( ax1.get_ylim() ) # Returns [min, max]
        #y_val_arr = clip( arr = y_val_arr, min = -5000, max = 5000 )

        #y_pos_arr = np.where( y_val_arr > 0, y_val_arr, 0 )#where(condition[, x, y]) -- where true, yield x, o.w. yield y
        #y_neg_arr = np.where( y_val_arr < 0, y_val_arr, 0 )#where(condition[, x, y]) -- where true, yield x, o.w. yield y

        y = np.concatenate( ( y0 , y1, y2, y3 ), axis = None )

        y_pos_arr = np.array( [ x for x in y if( x > 0 ) ] )
        y_neg_arr = np.array( [ x for x in y if( x < 0 ) ] )

        y_max = limit( arr = y_pos_arr, n = 3 )
        y_min = limit( arr = y_neg_arr, n = 3 )

        ax1.set_ylim( [ y_min, y_max ] )

        #=======================================================================

        ax2.plot( x_disc, [ x[ 3 ] for x in self.d_losses ], label='Gradient Penalty', color = 'magenta', linestyle = 'dashed', linewidth = 0.5 )

        ax2.set_title( 'Gradient Penalty', fontsize = 20 )
        ax2.set_xlabel('Batch Count (bpe: %s)' % bpe, fontsize = 14)
        ax2.set_ylabel('GP Loss', fontsize = 14)
        ax2.legend(loc='upper right', fontsize = 'large')

        #=======================================================================

        y_val_arr = np.array( ax2.get_ylim() )

        y_max = limit( arr = y_val_arr, n = 1 )
        y_min = y_val_arr.min()

        ax2.set_ylim( [ y_min, y_max ] )

        #=======================================================================

        plt.savefig( self.fig_path, edgecolor = 'black', linewidth = 15, figsize = figsize, bbox_inches='tight' )
        plt.close()

    def print_model_layers( self, architecture ):


        line_list = []
        get_str = lambda name, object : str( "\n" + name + ": " + str( object ) )
        new_line = lambda array, name, object : array.append( get_str( name = name, object = object ) )

        if( architecture == "parameters" ):
            header = "\nPARAMETERS"

            opening_header_block = header_1 + header + header_2 + "\n"
            closing_header_block = header_1 + "\n"

            line_list.append( opening_header_block )
            #print( opening_header_block )

            new_line( array = line_list, name = "Training Set", object = self.training_dir )
            new_line( array = line_list, name = "Batch Size", object = self.batch_size )
            new_line( array = line_list, name = "Optimizer", object = self.optimizer )

            new_line( array = line_list, name = "Critic Learning Rate", object = self.critic_learning_rate )
            new_line( array = line_list, name = "Critic Conv Filter Size", object = self.critic_conv_filters )
            new_line( array = line_list, name = "Critic Stride Size", object = self.critic_conv_strides )
            new_line( array = line_list, name = "Critic Kernel Size", object = self.critic_conv_kernel_size )

            new_line( array = line_list, name = "Generator Learning Rate", object = self.generator_learning_rate )
            new_line( array = line_list, name = "Generator Initial Dense Layer Size", object = self.generator_initial_dense_layer_size )
            new_line( array = line_list, name = "Generator Conv Filter Size", object = self.generator_conv_filters )
            new_line( array = line_list, name = "Generator Stride Size", object = self.generator_conv_strides )
            new_line( array = line_list, name = "Generator Kernel Size", object = self.generator_conv_kernel_size )
            new_line( array = line_list, name = "Generator Upsample", object = self.generator_upsample )

            line_list.append( str( "\n" + closing_header_block ) )
            #print( closing_header_block )

        else:
            if( architecture == "critic" ):
                header = "\nCRITIC ARCHITECTURE"
                model = self.critic

            else:
                header = "\nGENERATOR ARCHITECTURE"
                model = self.generator

            opening_header_block = header_1 + header + header_2 + "\n"
            closing_header_block = header_1 + "\n"

            line_list.append( opening_header_block )
            print( opening_header_block )

            for i in range( len( model.layers ) ):

                s = model.layers[ i ].get_output_shape_at( 0 )
                n = str( model.layers[ i ].name ).ljust( 16 )

                for( index ) in range( len( s ) ):

                    if( index == ( len( s ) - 1 ) ):
                        shape_str = shape_str.ljust( 19 )
                        new_string = str( s[ index ] ) + " )"
                        shape_str += str( new_string ).rjust( 8 )

                    elif( index >= 1 ):
                        new_string = str( s[ index ] ) + ","
                        shape_str += str( new_string ).rjust( 6 )

                    elif( index == 0 ):
                        new_string = "( " + str( s[ index ] ) + ","
                        shape_str  = str( new_string ).ljust( 7 )

                # Omit non-activation and up-sampling and batch-normalization layers
                if( n[ 0 : 5 ] != 'leaky'
                    and n[ 0 : 5 ] != 'up_sa'
                    and n[ 0 : 5 ] != 'batch'
                    and n[ 0 : 5 ] != 'tf_op'
                    and n[ 0 : 5 ] != 'insta' ):
                #if( n[ 0 : 5 ] != 'leaky' and n[ 0 : 5 ] != 'up_sa'and n[ 0 : 5 ] != 'batch' and n[ 0 : 5 ] != 'tf_op' ):
                #if( n[ 0 : 5 ] != 'leaky' and n[ 0 : 5 ] != 'up_sa'and n[ 0 : 5 ] != 'batch' ):

                    layer_string  = str( "Layer: %s"  % n ).ljust( 30 )
                    layer_string += str( "Output: %s" % shape_str )

                    line_list.append( str( "\n" + layer_string ) )
                    print( layer_string )

            line_list.append( str( "\n" + closing_header_block ) )
            print( closing_header_block )
        return( line_list )

    def print_model_summaries( self, critic = True, generator = True ):

        if( critic == True ): print( self.critic.summary(), '\n' )
        if( generator == True ): print( self.generator.summary(), '\n' )

    def clean_folders( self ) -> None:

        def clr_folder_files( file_list ):
            for filepath in file_list:
                 os.remove( filepath )

        path_check = lambda folder_dir : True if( os.path.isdir( folder_dir ) ) else( False )
        remove_files = lambda dir : clr_folder_files( glob.glob( dir + '/*' ) ) if( path_check( dir ) ) else( None )

        remove_files( self.results_dir + 'all' )
        remove_files( self.results_dir + 'gif_images/batch' )
        remove_files( self.results_dir + 'gif_images/single' )


    def clear_terminal( self ):
        os.system( 'clear' )

    def load_dataset( self, limit : int = None, augment : bool = False ):

        if( augment == True ):
            filename = ( "remy_images_%dx%d_augmented.npy" % ( self.img_size, self.img_size ) )
        else:
            filename = ( "remy_images_%dx%d.npy" % ( self.img_size, self.img_size ) )

        data_path = str( self.training_dir + filename )
        npy_data = np.load( data_path, allow_pickle = True )

        buffer_size = len( npy_data )
        npy_data = np.asarray( npy_data, dtype = np.float32 )

        #npy_data = npy_data.astype( 'float32' )
        npy_data = ( npy_data - 127.5 ) / 127.5 # Rescale: [-1,1]

        npy_data.reshape( buffer_size, self.img_size, self.img_size, 3 )
        np.random.shuffle( npy_data )

        if( limit ): npy_data = npy_data[ 0 : limit ]

        print( "\nDataset initialized, with shape: ", npy_data.shape )
        return( npy_data )

    def data2uint8( self, data ):
        output = data * 127.5 + 127.5                 # From [-1,1] to [0,255]
        output = np.array( output, dtype = 'uint8' )  # Cast img to uint8
        return( output )

    def generate_random_image( self ):

        fig = plt.figure( figsize = ( 4, 4 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )
        fig_path = self.results_dir + 'randomly_generated_image.png'

        random_noise = np.random.normal( 0, 1, ( 1, self.z_dim ) )
        generated_image = self.generator.predict( random_noise )#, training = False )
        decision = self.critic.predict( generated_image )#, training = False )

        #generated_image = self.generator( random_noise , training = False )
        #decision = self.critic( generated_image , training = False )

        val = decision[ 0 ][ 0 ]

        #print( "\nShape of generated image: ", generated_image[ 0 ].shape )
        #print( "critic decision: ", val )

        output = generated_image[ 0, :, :, : ]         # Numpy array object
        #print( "Output data type: ", output.dtype )    # Check data type: float 32
        gen_imgs = 0.5 * (output + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        #output = output * 255                          # Scale range to [0,255]
        #output = output.astype( 'uint8' )              # Cast img values to ints

        #output = output * 127.5 + 127.5                 # From [-1,1] to [0,255]
        #output = np.squeeze(np.round(output).astype(np.uint8))
        #output = np.array( output, dtype = 'uint8' )  # Cast img to uint8

        #output = self.data2uint8( output )
        ax = fig.add_subplot( 1, 1, 1 )
        ax.imshow( np.squeeze(gen_imgs), cmap = None, interpolation = 'nearest', vmin=0, vmax=1 ) # CHANGE_LATER
        #ax.imshow( np.squeeze(gen_imgs))# CHANGE_LATER

        ax.axis('off')

        fig.savefig( fig_path, bbox_inches = 'tight', quality = 100 )
        plt.close()

    def save_parameters( self, filename ):

        filepath = self.pickles_dir + filename + ".p"
        with open( filepath, 'wb' ) as f:
            pickle.dump( [
                self.gif,
                self.z_dim,
                self.img_size,
                self.color_chs,
                self.gif_rate,
                self.save_rate,
                self.input_dim,
                self.optimizer,
                self.batch_size,
                self.grad_weight,
                self.working_dir,
                self.training_dir,
                self.critic_conv_filters,
                self.critic_conv_kernel_size,
                self.critic_conv_strides,
                self.critic_batch_norm_momentum,
                self.critic_activation,
                self.critic_dropout_rate,
                self.critic_learning_rate,
                self.generator_initial_dense_layer_size,
                self.generator_upsample,
                self.generator_conv_filters,
                self.generator_conv_kernel_size,
                self.generator_conv_strides,
                self.generator_layer_normalization,
                self.generator_batch_norm_momentum,
                self.generator_activation,
                self.generator_dropout_rate,
                self.generator_learning_rate ],
                f
            )
        print( "\nModel parameters have been saved..." )



"""
    def _build_adversarial( self ):

        # INITIALIZE GENERATOR & critic
        self._build_generator()
        self._build_critic()

        #-----------------------------------------------------------------------
        # CONSTRUCT GENERATOR MODEL
        #-----------------------------------------------------------------------
        # generator_model is used to train generator layers...
        # --> thus critic layers must not be trainable
        # --> after model is compiled, changing .trainable will have no effect
        # --> as such, setting critic.trainable = True won't be an issue
        # --> as long as the generator_model is compiled first
        #-----------------------------------------------------------------------

        self.set_trainable_layers( model = self.critic, train = False )
        #self.set_trainable_layers( model = self.generator, train = True ) #???

        generator_input = tf.keras.layers.Input( shape = ( self.z_dim, ) ) # Fake image sample
        generator_layers = self.generator( generator_input )               # Returns fake img

        critic_layers_for_generator = self.critic( generator_layers )

        self.generator_model = tf.keras.models.Model(
            inputs = [ generator_input ],
            outputs = [ critic_layers_for_generator ],
        )

        self.generator_model.compile(
            optimizer = self.get_optimizer( lr = self.generator_learning_rate ),
            loss = self.wasserstein
        )

        #-----------------------------------------------------------------------
        # CONSTRUCT critic MODEL
        #-----------------------------------------------------------------------

        self.set_trainable_layers( model = self.critic, train = True )
        self.set_trainable_layers( model = self.generator, train = False )

        real_samples = tf.keras.layers.Input( shape = self.input_dim )

        generator_input_for_critic = tf.keras.layers.Input( shape = ( self.z_dim, ) )
        generator_samples_for_critic = self.generator( generator_input_for_critic )

        # Pass real & fake images through critic to calculate Wasserstein loss
        critic_output_from_generator = self.critic( generator_samples_for_critic )
        critic_output_from_real_samples = self.critic( real_samples )

        # Create interpolated image ( weighted average btwn real & fake images ), and pass through critic
        averaged_samples = RandomWeightedAverage( self.batch_size )( [ real_samples, generator_samples_for_critic ] )
        averaged_samples_out = self.critic( averaged_samples )

        # Use Python partial to pass interpolated images through the gradient penalty loss function
        partial_gp_loss = partial( self.gradient_penalty_loss, averaged_samples = averaged_samples, gradient_penalty_weight = self.grad_weight )
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function name


        self.critic_model = tf.keras.models.Model(
            inputs = [ real_samples, generator_input_for_critic ],
            outputs = [ critic_output_from_real_samples, critic_output_from_generator, averaged_samples_out ]
        )

        self.critic_model.compile(
            loss = [ self.wasserstein, self.wasserstein, partial_gp_loss ],
            optimizer = self.get_optimizer( lr = self.critic_learning_rate ),
        )
"""


"""
    def generate_random_image( self ):

        fig = plt.figure( figsize = ( 4, 4 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )
        fig_path = self.results_dir + 'randomly_generated_image.png'

        random_noise = np.random.normal( 0, 1, ( 1, self.z_dim ) )
        #generated_image = self.generator.predict( random_noise )#, training = False )
        #decision = self.critic.predict( generated_image )#, training = False )

        generated_image = self.generator( random_noise , training = False )
        decision = self.critic( generated_image , training = False )

        val = decision[ 0 ][ 0 ]

        #print( "\nShape of generated image: ", generated_image[ 0 ].shape )
        #print( "critic decision: ", val )

        output = generated_image[ 0, :, :, : ]         # Numpy array object
        #print( "Output data type: ", output.dtype )    # Check data type: float 32
        gen_imgs = 0.5 * (output + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        #output = output * 255                          # Scale range to [0,255]
        #output = output.astype( 'uint8' )              # Cast img values to ints

        #output = output * 127.5 + 127.5                 # From [-1,1] to [0,255]
        #output = np.squeeze(np.round(output).astype(np.uint8))
        #output = np.array( output, dtype = 'uint8' )  # Cast img to uint8

        #output = self.data2uint8( output )
        ax = fig.add_subplot( 1, 1, 1 )
        #ax.imshow( np.squeeze(gen_imgs), cmap = None, interpolation = 'nearest', vmin=0, vmax=1 ) # CHANGE_LATER
        ax.imshow( np.squeeze(gen_imgs))# CHANGE_LATER

        #ax.imshow( output, cmap = None, interpolation = 'nearest', vmin=0, vmax=255 )
        ax.axis('off')

        fig.savefig( fig_path, bbox_inches = 'tight', quality = 100 )
        plt.close()

    def generate_and_save_images( self, epoch = None, gnd_truth_img = None ):

        fig = plt.figure( figsize = ( 16, 16 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )

        all_images = self.results_dir + 'all/' + 'epoch_{:04d}.png'.format( self.epoch )
        passing_fakes = self.results_dir + 'passing_fakes/'
        new_image_batch = self.results_dir + 'recent.png'

        np.random.seed(1)
        sample_noise = np.random.normal( 0, 1, ( self.num_generated_imgs, self.z_dim ) )

        generated_images = self.generator.predict( sample_noise )
        decisions = self.critic.predict( generated_images )

        prediction_msg = ""
        decision_arr = []
        index_arr = []
        pos_cnt = 0

        # Rescale images from 0 - 1
        gen_imgs = 0.5 * (generated_images + 1) # From [-1,1] to [0,1]
        gen_imgs = np.clip(gen_imgs, 0, 1)      # Make sure vals are between 0,1

        for i in range( len( generated_images ) ):

            label = decisions[ i ]
            ax = fig.add_subplot( 4, 4, i + 1 )
            img = np.squeeze( gen_imgs[ i, :, :, : ] ) # Squeeze down to 3 dim

            #ax.imshow( img, cmap = None, interpolation = 'nearest', vmin = 0, vmax = 1 ) # CHANGE_LATER
            ax.imshow( img, cmap = None, interpolation = 'nearest' ) # CHANGE_LATER
            #ax.imshow( img )# CHANGE_LATER

            ax.set_title( label )
            ax.axis( 'off' )

            if( label > 0 ):

                index_arr.append( i )
                decision_arr.append( label )
                pos_cnt += 1

        fig.savefig( all_images, bbox_inches = 'tight', quality = 90 ) #linewidth = 15, edgecolor = 'black',
        fig.savefig( new_image_batch, bbox_inches = 'tight', quality = 90 ) #linewidth = 15, edgecolor = 'black',

        if( self.gif == True and epoch % self.gif_rate == 0 ):

            gif_image_path = self.gif_images_path + 'image_{:07d}.png'.format( self.epoch )
            fig.savefig( gif_image_path, bbox_inches = 'tight', quality = 90 ) #linewidth = 15, edgecolor = 'black',

            self.gif_creator.set()

        plt.close()

        if( pos_cnt > 0 ):

            decision_arr_index = np.argmax( decision_arr )
            generated_img_index = index_arr[ decision_arr_index ]

            img = self.data2uint8( generated_images[ generated_img_index ] )
            val = min( decision_arr )[ 0 ]
            #val = min( decision_arr )[ 0 ]

            #gt_img = np.expand_dims( gnd_truth_img, axis = 0 )
            #gnd_truth_decision = self.critic.predict( gt_img )[ 0 ][ 0 ]
            #print( "GroundTruth: ", gnd_truth_decision )

            #print( "critic_decision_{}: ".format( self.epoch ), val )
            prediction_msg = "[Critic_Prediction: %s]" % ( self.num2str( string = val, round_val = 2, just_val = 7, just_dir = 'r' ) )
            prediction_msg += "[Pos_Cnt: %s]" % ( self.num2str( string = pos_cnt, round_val = 0, just_val = 2, just_dir = 'r' ) )
            #prediction_msg += "[GT: %s]" % ( self.num2str( string = gnd_truth_decision, round_val = 2, just_val = 7, just_dir = 'r' ) )

            fig = plt.figure( figsize = ( 4, 4 ), facecolor = 'white', linewidth = 5, edgecolor = 'black' )
            fig_path = self.results_dir + 'passing_fakes/'
            fig_path += 'epoch_{:04d}'.format( self.epoch )
            fig_path += '_with_decision_score_' + str( val ) + '.png'

            ax = fig.add_subplot( 1, 1, 1 )
            ax.imshow( img, cmap = None, interpolation = 'nearest' )
            ax.set_title( val )
            ax.axis('off')

            fig.savefig(fig_path, bbox_inches = 'tight', quality = 100)#linewidth = 15, edgecolor = 'black',
            plt.close()

        gt_img = np.expand_dims( gnd_truth_img, axis = 0 )
        gnd_truth_decision = self.critic.predict( gt_img )[ 0 ][ 0 ]
        #print( "GroundTruth: ", gnd_truth_decision )

        prediction_msg += "[GT: %s]" % ( self.num2str( string = gnd_truth_decision, round_val = 2, just_val = 7, just_dir = 'r' ) )

        return( prediction_msg )
"""
