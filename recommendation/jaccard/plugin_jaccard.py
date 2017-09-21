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

from __future__ import division

def intersection(seq1, seq2):
    """
    Intersection between two sequences. Items that appears in both sequences.
    """
    
    return list(set(seq1) & set(seq2))

def union(seq1, seq2):
    """
    Union between two sequences. All items that appears in the two sequeces.
    """
    
    return list(set(seq1 + seq2))

def distance(seq1, seq2, extra={}):
    """
    Jaccard similarity between two sequences.

    Jaccard similarity is defined by the division between the items that
    appears in both sequences and the items presents in the two sequences.
    """
    
    return 1 - (len(intersection(seq1, seq2)) / len(union(seq1, seq2)))

def get_text():
    return "Jaccard"
