__author__ = 'said'

import time
import environment

class CSVEnvironment(environment.Environment):

    stateLength = 10
    timeOffset = 60
    statesInQueue = 0

    def __init__(self):
        super(CSVEnvironment, self).__init__()

    def run(self):
        i = 0
        while( self.mustStop() == False ):
            print "CSV... " + str(i) + " \n"
            time.sleep(0.25)
            i += 1

    def getStateLength(self):
        return self.stateLength

    def getState(self):
        return None



