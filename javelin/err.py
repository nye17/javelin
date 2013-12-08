#Last-modified: 04 Dec 2013 16:21:16

all = ['InputError', 'UsageError', 'Error']

""" Error handlers. 
"""

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




