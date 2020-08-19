import os
import warnings
import tensorflow as tf
from tensorflow.python.util import deprecation

warnings.filterwarnings( "ignore" )
#os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
deprecation._PRINT_DEPRECATION_WARNINGS = False

class TF_Environment():
    def __init__( self, gpu_fraction = 0.3, gpu_growth = True ):

        #tf.python.keras.backend.set_learning_phase(0) # inference mode
        #tf.python.keras.backend.set_learning_phase(1) # training mode

        #tf.compat.v1.experimental.output_all_intermediates( True )

        tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )
        tf.compat.v1.disable_eager_execution()
        tf.keras.backend.clear_session()

        self.gpu_fraction = gpu_fraction
        self.gpu_growth = gpu_growth
        self._set_config()

    def _set_config( self ):
        #self.config = tf.compat.v1.ConfigProto( log_device_placement=True )
        self.config = tf.compat.v1.ConfigProto()

        #self.config.gpu_options.visible_device_list = "0"
        self.config.gpu_options.allow_growth = self.gpu_growth
        self.config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction

    def run_session( self ):
        sess = tf.compat.v1.Session( config = self.config )
        print( sess )
        self.session = sess

    def end_session( self ):
        self.session.close()
