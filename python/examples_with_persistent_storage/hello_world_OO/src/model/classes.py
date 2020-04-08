class hello(object):

    def __init__(self, message):
        super(hello, self).__init__()
        self.message = message

    def get(self):
        return self.message
