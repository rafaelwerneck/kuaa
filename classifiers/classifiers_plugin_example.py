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

#Python imports

#Framework imports

def classify(images, train_set, test_set, fv_pos, descriptor, parameters):
    """
    Performs the classification of the test_set according to the train_set.
    """
    
    print "Classification: EXAMPLE"
    
    #Get parameters
    
    #Read the train file and save the list of class and the list
    #of feature vectors
    list_class = []
    list_fv = []
    
    for img in train_set:
        list_class.append(images[img][0])
        list_fv.append(images[img][1][fv_pos])
    
    #Classification
    #--------------------------------------------------------------------------
    #Preprocess the list of class to integers
    label_encoder = preprocessing.LabelEncoder()
    list_class = label_encoder.fit_transform(list_class)
    print "List of classes of this experiment:", label_encoder.classes_
    
    #Save configuration of the EXAMPLE
    str_configuration = None
    
    list_img = test_set
    list_class = []
    list_fv = []
    
    for img in test_set:
        list_class.append(images[img][0])
        list_fv.append(images[img][1][fv_pos])
    
    list_result = []
    #--------------------------------------------------------------------------
    
    return list_img, list_class, list_result, label_encoder.classes_, \
           str_configuration
