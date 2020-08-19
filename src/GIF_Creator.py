import glob
import imageio

class GIF_Creator():

    def __init__( self, filepath, imgspath ):
        self.filepath = filepath
        self.imgspath = imgspath

    def set( self ):

        with imageio.get_writer( self.filepath, mode= 'I' ) as writer:

          filenames = glob.glob( str( self.imgspath + '*' ) )
          filenames = sorted(filenames)
          last = -1

          for i,filename in enumerate(filenames):
            #frame = 2*(i**0.5)
            frame = i

            if round(frame) > round(last): last = frame
            else: continue

            image = imageio.imread(filename)
            writer.append_data(image)

          image = imageio.imread(filename)
          writer.append_data(image)
