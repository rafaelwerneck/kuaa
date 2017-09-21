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

#install jellyfish (https://pypi.python.org/pypi/jellyfish/)

from sklearn import preprocessing
import jellyfish

INDEX_ALPHA = 97

def pos_to_char(pos):
    """
    Transforms the unique label of the preprocessing as letters.
    """
    
    return chr(pos + INDEX_ALPHA)

def distance(seq1, seq2, extra={}):
    """
    Jaro-Winkler distance, according to the matchings between the items in the
    sequence and the transposed item in the sequences. Later it favorates the
    substring that matches from the beginning of the sequences.
    """
    
    #Preprocessing
    label_processing = preprocessing.LabelEncoder()
    label_processing.fit(seq1 + seq2)
    
    #Preprocess seq1
    label_seq1 = label_processing.transform(seq1)
    alpha_seq1 = ''.join([pos_to_char(item_label) for item_label in
        label_seq1])
    
    #Preprocess seq2
    label_seq2 = label_processing.transform(seq2)
    alpha_seq2 = ''.join([pos_to_char(item_label) for item_label in
        label_seq2])
    
    #Jaro-Winkler distance
    return 1 - jellyfish.jaro_winkler(alpha_seq1, alpha_seq2)

def get_text():
    return "Jaro-Winkler"
