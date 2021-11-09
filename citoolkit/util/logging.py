""" Contains the CIToolkit logging function "cit_log" used to pretty print verbose output """

import time

def cit_log(text):
    """ Pretty prints log text."""
    assert isinstance(text, str)

    timestamp = time.strftime("%d %b %Y, %H:%M:%S")
    print("(CIToolkit) " + timestamp + ": " + text)
