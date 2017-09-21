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

import os
import sys
import heapq
import random
import copy

KEY = 0
DISTANCE = 1
NUM_TRAIN_QUERIES = 20
RECOMMENDED_DICT = 1

def main(train_dict, test_sequence, recommend_radio, number_recommendations):
    """
    Main function responsible to calling the plugins 
    """
    
    #Path to the train queries
    recommendation_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recommendation results"))
    
    #Add path to distance plugin
    distance_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recommendation", recommend_radio))
    if distance_path not in sys.path:
        sys.path.append(distance_path)
    
    software = __import__("plugin_" + recommend_radio)
    print software
    
    extra = {}
    if recommend_radio.find("tfidf") == 0:
        list_exp = train_dict.values()
        idf_matrix = software.idf(list_exp)
        extra["idf_matrix"] = idf_matrix
        
    distances = []
    
    if recommend_radio == "lrar":
        distances = software.distance(train_dict, test_sequence, recommendation_folder)
        
        distances_sorted = heapq.nlargest(number_recommendations, distances, key=lambda x: x[DISTANCE])
    else:
        for train_key, train_sequence in train_dict.iteritems():
            distances.append((train_key, software.distance(train_sequence, test_sequence, extra)))
    
        distances_sorted = heapq.nsmallest(number_recommendations, distances, key=lambda x: x[DISTANCE])
    
    return distances_sorted
