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

#Future
from __future__ import division #import to change the operation '/' to return 
                                #a float

#Python imports
import os
from sklearn.feature_extraction.text import TfidfTransformer

#Framework imports
    
#CONSTANTS
POS_CLASSES = 0
POS_FV = 1
ZERO_INDEX = 0

def normalize(img_path, images, images_set, pos_train_test, parameters, method,
        train_param):
    """
    Function that performs the normalization of a feature vector.
    
    For each item in the feature vector, calculate the term frequency of this 
    item in the feature vector, and return the feature with the term frequency
    of an item in the place of that item.
    """
    
    #Get parameters
    tfidf_norm = norm(parameters['Norm'])
    
    print "Normalizer: TERM FREQUENCY"
    
    #Get the list of classes and the feature vector of the img_path
    img_classes = images[img_path][POS_CLASSES]
    try:
        img_fv = images[img_path][POS_FV][pos_train_test]
    except:
        img_fv = images[img_path][POS_FV][ZERO_INDEX]
    
    print "\tFeature vector of image", img_path, \
            "being normalized by process", os.getpid()
    
    # Performs the normalization --------------------------------------------- 
    transformer = TfidfTransformer(norm=tfidf_norm, use_idf=False)
    fv_norm = transformer.transform(img_fv).toarray().tolist()[0]
    #-------------------------------------------------------------------------
    
    return img_path, len(img_classes), img_classes, fv_norm, train_param

def norm(norm_name):
    """
    Assign a value to each name of norm possible to normalize the feature.
    """
    
    if norm_name == 'Manhattan':
        return 'l1'
    elif norm_name == 'Euclidean':
        return 'l2'
    else:
        return None
