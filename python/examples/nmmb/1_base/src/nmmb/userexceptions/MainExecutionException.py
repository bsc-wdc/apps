
class MainExecutionException(Exception):

    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(MainExecutionException, self).__init__(message)
