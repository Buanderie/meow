__author__ = 'said'

import environment

class CSVEnvironment(environment.Environment):

    stateLength = 10
    timeOffset = 60

    def run(self):
        while(1):
            print "CSV...\n"

    def getStateLength(self):
        return self.stateLength

    def getState(self):
        return None



