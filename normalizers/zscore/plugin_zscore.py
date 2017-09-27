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
from __future__ import division

#Python imports
import os
import numpy
from scipy import stats
    
#CONSTANTS
POS_CLASSES = 0
POS_FV = 1
ZERO_INDEX = 0

def normalize(img_path, images, images_set, pos_train_test, parameters, method,
        train_param):
    """
    Function that performs the normalization of a feature vector.
    
    Calculates the z-score of each position in the feature vector, relative to
    the sample mean and standard deviation of that position in all feature
    vectors.
    """

    print "Normalizer: ZSCORE"
    
    #Get the list of classes and the feature vector of the img_path
    img_classes = images[img_path][POS_CLASSES]
    try:
        img_fv = images[img_path][POS_FV][pos_train_test]
    except:
        img_fv = images[img_path][POS_FV][0]

    print "\tFeature vector of image", img_path, \
          "being normalized by process", os.getpid()

    # Performs the normalization ---------------------------------------------
    #If the parameters of normalization don't exists, calculate the mean and
    #   the standard deviation of the feature vectors in the train set
    if 'Mean' not in train_param:
        list_train = []
        for image in images_set:
            try:
                list_train.append(images[image][POS_FV][pos_train_test])
            except:
                list_train.append(images[image][POS_FV][ZERO_INDEX])
        
        mean_list = numpy.mean(list_train, axis=0)
        std_list = numpy.std(list_train, axis=0)
        
        train_param['Mean'] = mean_list
        train_param['Deviation'] = std_list
    #If the parameters of normalization already exists, load them
    else:
        print "\t\tGet Mean and Standard Deviation"
        mean_list = train_param['Mean']
        std_list = train_param['Deviation']
    
    fv_norm = [(img_fv[index] - mean_list[index]) / std_list[index]
            for index in range(len(img_fv))]
    fv_norm = [fv_item for fv_item in fv_norm if not numpy.isnan(fv_item)]
    #-------------------------------------------------------------------------

    return img_path, len(img_classes), img_classes, fv_norm, train_param
