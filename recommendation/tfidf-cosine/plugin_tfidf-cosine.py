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

from sklearn.feature_extraction import text
from sklearn.metrics import pairwise

INDEX_ZERO = 0

def idf(list_seq):
    """
    Calculates the Inverse Document Matrix of the list of sequences.
    """
    
    idf_matrix = text.TfidfVectorizer()
    
    list_string_seq = [" ".join(seq) for seq in list_seq]
    
    idf_matrix.fit(list_string_seq)
    
    return idf_matrix
    
def distance(seq1, seq2, extra={}):
    """
    Performs TF-IDF on the sequences and calculates the euclidean distance
    between them.
    """
    
    idf_matrix = extra["idf_matrix"]
    
    #Seq1
    string_seq1 = " ".join(seq1)
    seq1_fv = idf_matrix.transform([string_seq1]).todense()
    
    #Seq2
    string_seq2 = " ".join(seq2)
    seq2_fv = idf_matrix.transform([string_seq2]).todense()
    
    #Distance
    return pairwise.pairwise_distances(seq1_fv, seq2_fv, 'cosine')[INDEX_ZERO][INDEX_ZERO]

def get_text():
    return "TF-IDF Cosine"
