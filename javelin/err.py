#Last-modified: 06 Mar 2012 12:02:02 PM

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        msg  -- explanation of the error
    """
    def __init__(self, msg):
        self.msg = msg
        print(msg)


class UsageError(Error):
    """Exception raised for errors in the usage of methods.

    Attributes:
        msg  -- explanation of the error
    """
    def __init__(self, msg):
        self.msg = msg
        print(msg)




