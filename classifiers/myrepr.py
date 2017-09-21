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

import types

def myrepr(obj):
    """
    """
    if isinstance(obj, types.FunctionType):
        return obj.func_name + '()'
    elif isinstance(obj, list):
        reprs = map(myrepr, obj)
        reprs = reduce(lambda x, y: x + ',' + y, reprs)
        return '[' + reprs + ']'
    elif isinstance(obj, tuple):
        if len(obj) == 0:
            return '()'
        reprs = map(myrepr, obj)
        reprs = reduce(lambda x, y: x + ',' + y, reprs)
        reprs += ',' if len(obj) == 1 else ''
        return '(' + reprs + ')'
    elif isinstance(obj, set):
        reprs = myrepr(list(obj))
        return 'set(' + reprs + ')'
    elif isinstance(obj, dict):
        if len(obj) == 0:
            return '{}'
        reprs = zip(map(myrepr, obj.keys()), map(myrepr, obj.values()))
        reprs = map(lambda (k,v): k + ':' + v, reprs)
        reprs = reduce(lambda x, y: x + ',' + y, reprs)
        return '{' + reprs + '}'
    elif isinstance(obj, type):
        return repr(obj).replace(' ', '')
    else:
        return repr(obj)
