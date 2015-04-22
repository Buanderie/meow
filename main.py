"""Module docstring.

This serves as a long usage message.
"""
import sys
import getopt
import time

from csvenvironment import *

def main():
    popo = CSVEnvironment()
    print popo.getStateLength()
    popo.start()

    time.sleep(3)
    popo.stop()

if __name__ == "__main__":
    main()