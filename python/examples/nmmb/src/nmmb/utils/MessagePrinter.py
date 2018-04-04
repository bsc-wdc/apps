class MessagePrinter(object):
    """
    Helper class to print messages
    """

    # For info messages
    LINE_SIZE = 60
    PRE_CHARS_HEADER_LINE = 11
    PRE_CHARS_MSG_LINE = 5

    # Loggers
    logger = None

    def __init__(self, logger):
        """
        Creates a message printer for a specific logger
        :param logger: Logger to use
        """
        self.logger = logger

    def printHeaderMsg(self, msg):
        """
        Prints msg as header message
        :param msg: Message to be printed as header
        """
        # Separator line
        self.logger.info("")

        # Message line
        sb = "========= "
        sb += msg
        sb += " "
        for i in range(self.PRE_CHARS_HEADER_LINE + len(msg), self.LINE_SIZE):
            sb += "="

        self.logger.info(sb)

    def printInfoMsg(self, msg):
        """
        Prints msg as print info message
        :param msg: Message to be printed as info
        """
        # Separator line
        self.logger.info("")

        # Message line
        sb = "--- "
        sb += msg
        sb += " "
        for i in range(self.PRE_CHARS_HEADER_LINE + len(msg), self.LINE_SIZE):
            sb += "-"

        self.logger.info(sb)
