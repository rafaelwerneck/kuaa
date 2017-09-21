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
cyan_err('import Classifier')

import copy
import numpy as np
from myrepr import myrepr


class Classifier:
    """
    To implement a classifier it is necessary to implement the
    following functions: (1) 'simple__init__(self)', (2)
    'simple__repr__(self)', (3) 'simple__gridsearch__(self)', (4)
    'simplefit(self)', and (5) 'simpleclassify(self)'.

    It is necessary to implement the 'simplefit()' function
    assuming the training features are in 'self._vftr' and the
    label of the corresponding features are in 'self._vltr'.  The
    'Classifier.simplefit(self)' can be called as a form of
    assertion.

    It is necessary to implement the 'simpleclassify()' function
    assuming the test features are in 'self._vfte' and the label
    of the corresponding features are in 'self._vlte'.  The
    'Classifier.simpleclassify(self)' can be called as a form of
    assertion.

    To use the classifier, it is used only the 'fit(self, vftr,
    vltr)' and 'classify(self, vfte, vlte)' functions, where
    'vftr', 'vltr', 'vfte', and 'vlte' are the vector (list) of
    training feature, the vector of training labels, the vector
    of testing feature, and the vector of testing labels,
    respectively.

    In fitting step (in the call of 'fit(self, vftr, vltr)'), the
    'simple__gridsearch__(self)' function is called if
    'self._gridsearch' is defined 'False' in 'simple__init__()'
    function.  It is advisable to call
    'Classifier.simple__gridsearch__(self)' in the beggining of a new
    implementation of 'simple__gridsearch__(self)' as a form of
    assertion.
    """
    def __save__(self, path):
        self.simple__save__(path)

    def __load__(self, path):
        self.simple__load__(path)

    def simple__load__(self, path):
        assert False

    def simple__save__(self, path):
        assert False

    def __init__(self, **parameters):
        """
        A '__init__()' function must set: (1) 'self._trained' to
        'False' and (2) 'self._parameters' to 'None' or some
        other value.

        A implementation of a new classifier probably will call
        'Classifier.__init__(self, parameters)' to set
        'self._trained' to 'False' and 'self._parameters' to
        'parameters'.
        """
        self._trained = False
        self._parameters = parameters

        assert isinstance(self._parameters, dict)

        self.simple__init__()

        assert hasattr(self, '_gridsearch')
        assert isinstance(self._gridsearch, bool)

    def __repr__(self):
        return self.simple__repr__() + ' & % {0}'.format(myrepr(self._parameters))

    def __gridsearch__(self):
        """
        See doc of 'Classifier' class.
        """
        self.simple__gridsearch__()

    def config(self, **parameters):
        """
        Sets any parameter passed to this function in this classifier's
        self._parameters.
        """
        assert isinstance(parameters, dict)
        self._parameters.update(parameters)

    def simpleconfig(self):
        """
        This fuction is called whenever config() is called. Here there should
        be asserts for the new configuration values.
        """
        pass

    def fit(self, vftr, vltr):
        """
        The set of training feature must contain no sample of a
        unknown class.

        See doc of 'Classifier' class.
        """
        assert not self._trained

        self._vftr = vftr.tolist() if isinstance(vftr, np.ndarray) else copy.deepcopy(vftr)
        self._vltr = vltr.tolist() if isinstance(vltr, np.ndarray) else copy.deepcopy(vltr)

        assert isinstance(self._vftr, list)
        assert isinstance(self._vltr, list)
        assert len(self._vftr) == len(self._vltr)
        assert len(set(map(len, self._vftr))) == 1

        self._acs = list(set(filter(lambda label: label is not None,
                                    self._vltr)))
        self._unknown_label = min(self._acs) - 999

        assert self._unknown_label not in self._acs
        assert None not in self._vltr

        if self._gridsearch:
            self.__gridsearch__()
        self.simplefit()

        del self._vftr
        del self._vltr

        self._trained = True
        magenta_err((self, 'fitted'))

    def classify(self, vfte, vlte=None, auxiliar=None):
        """
        The unknown testing labels must be 'None'.

        See doc of 'Classifier' class.
        """
        assert self._trained

        self._vfte = vfte.tolist() if isinstance(vfte, np.ndarray) else copy.deepcopy(vfte)
        if vlte is not None:
            self._vlte = vlte.tolist() if isinstance(vlte, np.ndarray) else copy.deepcopy(vlte)
        else:
            self._vlte = [None] * len(self._vfte)

        assert isinstance(self._vfte, list)
        assert isinstance(self._vlte, list)
        assert len(self._vfte) == len(self._vlte)
        assert len(set(map(len, self._vfte))) == 1

        assert self._unknown_label not in self._vlte
        assert all([label in self._acs
                    for label in self._vlte
                    if label is not None])

        ret = self.simpleclassify(auxiliar)

        del self._vfte
        del self._vlte

        assert self._unknown_label not in ret
        assert all([label in self._acs
                    for label in ret
                    if label is not None])

        return ret

    def copy(self):
        """
        See doc of 'Classifier' class.
        """
        ret = copy.deepcopy(self)

        return ret

    def simple__init__(self):
        """
        Every classifier must define a 'simple__init__(self)' function
        and assign a boolean value to 'self._gridsearch'.
        """
        assert isinstance(self._parameters, dict)

    def simple__repr__(self):
        assert False

    def simple__gridsearch__(self):
        """
        See doc of 'Classifier' class.
        """
        assert self._gridsearch

    def simplefit(self):
        """
        See doc of 'Classifier' class.
        """
        assert not self._trained
        assert self._vftr
        assert self._vltr
        assert self._unknown_label not in self._acs

    def simpleclassify(self, auxiliar=None):
        """
        See doc of 'Classifier' class.

        This function must return 'None' for unknown samples.
        """
        assert self._trained
        assert self._vfte
        assert self._vlte
