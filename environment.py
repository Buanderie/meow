__author__ = 'said'

from threading import *

class Environment(Thread):

    hasToStop = False
    lock = Lock()

    def mustStop(self):
        ret = False
        self.lock.acquire()
        ret = self.hasToStop
        self.lock.release()
        return ret

    def stop(self):
        self.lock.acquire()
        self.hasToStop = True
        self.lock.release()

    def getStateLength(self):
        return None

    def getState(self):
        return None

