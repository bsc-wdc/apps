
class CommandException(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(CommandException, self).__init__(message)
