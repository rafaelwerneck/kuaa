#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kuaa.
#
# Kuaa is a framework for the automation of machine learning experiments.
#
# It provides a workflow-based standardized environment for easy evaluation of
# feature descriptors, normalization techniques, classifiers and fusion
# approaches.
#
# Techniques of each kind can be easily plugged into the framework as they can
# be implemented as plugins, with standardized inputs and outputs.
# The framework also provides a recommendation module in order to help
# inexperienced researchers in choosing adequate or alternative techniques for
# experiments.
#
# Copyright (C) 2016 under the GNU General Public License Version 3.
#
# This framework was developed during the research collaboration of Institute
# of Computing (University of Campinas, Brazil) and Samsung Eletrônica da
# Amazônia Ltda. entitled "Pattern recognition and classification by feature
# engineering, *-fusion, open-set recognition, and meta-recognition", which was
# sponsored by Samsung.
#
# This framework is provided "as is" without any guarantees or warranty. The
# authors make no warranties, express of implied, that they are free of error,
# or they will meet your requirements for any particular application.
#
# The framework was developed to be used for educational and research purposes.
# It is expressly prohibited to use for any commercial purposes.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

from termcolor import *
import sys
cyan_err('import hashfile')

import types
import numpy as np
from random import random
import os
import multiprocessing as mp
import signal


def no_signal_handler(signal, frame):
    red_err('You pressed Ctrl+C in hashfile.py! Try again later!')


class StorageFile():
    def __init__(self, hashname):
        """
        """
        assert isinstance(hashname, str)
        assert hashname[-4:] == '.npz'

        self.__hashname = hashname

    def __repr__(self):
        return 'StorageFile({0})'.format(self.__hashname)

    def saveVars(self, **kwargs):
        """
        """
        assert len(kwargs) > 0

        vartypes = [(k, type(v))
                        for k, v
                        in kwargs.iteritems()]
        np.savez(self.__hashname, vartypes=vartypes, **kwargs)

    def loadVars(self, *args):
        """
        """
        assert len(args) > 0

        try:
            data = np.load(self.__hashname)
            vartypes = dict(map(tuple, data['vartypes']))
            ret = [np.array(data[arg])
                     if vartypes[arg] == np.ndarray
                     else vartypes[arg](data[arg]) for arg in args]
        except:
            raise Exception

        return ret if len(ret) > 1 else ret[0]

class HashFile():
    __lock = mp.Lock()

    def __repr__(self):
        print 'HashFile: ', self.__hashfile

    def __init__(self, storagedir, hashfile):
        """
        """
        assert isinstance(storagedir, str)
        assert isinstance(hashfile, str)
        assert storagedir[-1] == '/'
        assert hashfile[-4:] == '.npz'
        assert hashfile.find('/') == -1

        self.__storagedir = storagedir
        self.__hashfile = storagedir + hashfile

    def getStorageFile(self, params):
        """
        Input:
            prefixname :: String
            hashnamekey :: String

        If you change the function 'prefixname', then it is safe to
            remove all files will prefix 'prefixname' stored in 'storagedir'
            to ensure a correct results according to the changes.

        Limitations:
            It is not useful in a nondeterministic function.
            Take care with functions that uses global parameters.
        """
        assert isinstance(params, list)

        signal.signal(signal.SIGINT, no_signal_handler)
        HashFile.__lock.acquire()
        try:
             callersname = sys._getframe(1).f_code.co_name
             hashnamekey = self.__createHashNameKey(callersname, params)

             self.__hashtable = self.loadHashTable()
             yellow_err('HashFile.getStorageFile(): size: {0}'.format(len(self.__hashtable)))
             try:
                 hashname = self.__hashtable[hashnamekey]
             except:
                 self.__hashtable[hashnamekey] = self.createHashName(callersname)
                 while len(self.__hashtable) != len(set(self.__hashtable.values())):
                     """
                     To ensure the created hash name do not exists previously.

                     It is very possible this part of the code will never run.
                     """
                     self.__hashtable[hashnamekey] = self.createHashName(callersname)
                 self.__saveHashTable()
                 hashname = self.__hashtable[hashnamekey]

             return StorageFile(self.__storagedir + hashname + '.npz')
        finally:
            HashFile.__lock.release()
            signal.signal(signal.SIGINT, signal.default_int_handler)

    def removeHashTable(self):
        """
        """
        os.remove(self.__hashfile)

    @staticmethod
    def createHashName(prefixname=''):
        """
        """
        assert isinstance(prefixname, str)

        ret = (prefixname + '-'
                 if len(prefixname) > 0
                 else '') + str(random())[2:] + str(random())[2:] + str(random())[2:] + str(random())[2:]

        assert isinstance(ret, str)

        return ret

    def loadHashTable(self):
        """
        """
        try:
            hashtable = np.load(self.__hashfile)
            hashtable = dict(map(lambda (x, y): (x, str(y)),
                                        dict(hashtable).iteritems()))
        except:
            hashtable = {}

        return hashtable

    def __saveHashTable(self):
        """
        """
        try:
            np.savez(self.__hashfile, **self.__hashtable)
        except:
            raise Exception('Possibly, the arguments to the getStorageFile function are very big.')

    @staticmethod
    def __myrepr(obj):
        """
        """
        if isinstance(obj, types.FunctionType):
            return obj.func_name + '()'
        elif isinstance(obj, list):
            reprs = map(HashFile.__myrepr, obj)
            reprs = reduce(lambda x, y: x + ',' + y, reprs)
            return '[' + reprs + ']'
        elif isinstance(obj, tuple):
            if len(obj) == 0:
                return '()'
            reprs = map(HashFile.__myrepr, obj)
            reprs = reduce(lambda x, y: x + ',' + y, reprs)
            reprs += ',' if len(obj) == 1 else ''
            return '(' + reprs + ')'
        elif isinstance(obj, set):
            reprs = HashFile.__myrepr(list(obj))
            return 'set(' + reprs + ')'
        elif isinstance(obj, dict):
            reprs = zip(map(HashFile.__myrepr, obj.keys()), map(HashFile.__myrepr, obj.values()))
            reprs = map(lambda (k,v): k + ':' + v, reprs)
            reprs = reduce(lambda x, y: x + ',' + y, reprs)
            return '{' + reprs + '}'
        elif isinstance(obj, type):
            return repr(obj).replace(' ', '')
        else:
            return repr(obj)

    @staticmethod
    def __createHashNameKey(callersname, params):
        """
        """
        assert isinstance(callersname, str)
        assert isinstance(params, list)

        reprs = [str(hash(HashFile.__myrepr(par))) for par in params] # PEDRORMJUNIOR 20130806
        reprs = reduce(lambda x, y: x + '|' + y, reprs)
        hashnamekey = callersname + '()|' + reprs

        return hashnamekey
