class FileObject():

    def __init__( self, filepath ):
        self.filepath = filepath

    def _open_file( self ):
        self.file = open( self.filepath, "w+" ) # 'w+' - write & read, overwriting or creating new file

    def _close_file( self ):
        self.file.close()

    def write2file( self, line_list, open_file = False, close_file = False ):
        if( open_file == True ): self._open_file()
        self.file.writelines( line_list )
        if( close_file == True ): self._close_file()
