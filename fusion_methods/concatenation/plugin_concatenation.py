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
    
#CONSTANTS
POS_CLASSES = 0
POS_FV = 1
INDEX_ZERO = 0

def fusion(list_images, list_classes, list_train_test, classes_list,
        result_path, parameters):
    """
    Performs the fusion of the feature vectors.
    
    For each image of the database, concatenate the feature vectors of each
    input.
    """
    
    #Get parameters
    
    images = {}
    
    #Fusion Method
    #-------------------------------------------------------------------------
    #As originated from the same database, every classes dictionary is equal
    classes = list_classes[INDEX_ZERO]
    #As the concatenation is made in feature vectors in the same train and
    # test set, return the first of the list
    train_test = list_train_test[INDEX_ZERO]
    
    len_fusion = len(list_train_test)
    for index_fusion in range(len_fusion):
        train_test = list_train_test[index_fusion]
        len_train_test = len(train_test)
        for index_train_test in range(len_train_test):
            for img in list_images[index_fusion].iterkeys():
                try:
                    img_fv = list_images[index_fusion][img][POS_FV][index_train_test]
                except:
                    img_fv = list_images[index_fusion][img][POS_FV][INDEX_ZERO]
                
                if img not in images.keys():
                    img_classes = list_images[index_fusion][img][POS_CLASSES]
                    temp_fv = []
                    for i in range(len_train_test):
                        temp_fv.append([])
                    images[img] = [img_classes, temp_fv]
                images[img][POS_FV][index_train_test].extend(img_fv)
    #-------------------------------------------------------------------------
    print "Success in the concatenation."
    
    print "Saving the result of the Concatenation into files"
    save_file(result_path, images, classes, train_test)
    
    return images, classes, train_test

def save_file(result_path, images, classes, train_test):
    """
    Save the result of the concatenation into a file.
    """
    
    for index, train_test_set in enumerate(train_test):
        train, test = train_test_set
        result_file = open(result_path + str(index) + ".txt", "wb")
        for image_path in train + test:
            image_classes = images[image_path][POS_CLASSES]
            image_predict = images[image_path][POS_FV][index]
            result_file.write(image_path + " " + str(len(image_classes)) + \
                    " " + str(image_classes) + " " + str(image_predict) + "\n")
        result_file.close()
