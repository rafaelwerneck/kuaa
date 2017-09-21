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
from numpy import zeros, array, where

#Framework imports
    
#Constants
POS_CLASSES = 0
POS_PREDICT = 1
INDEX_ZERO = 0
INDEX_TRAIN = 0
INDEX_TEST = 1

def fusion(list_images, list_classes, list_train_test, classes_list,
        result_path, parameters):
    """
    Performs the fusion of the classes predicted.
    
    For each image of the database, choose the predict class that is more
    representative.
    """
    
    #Get parameters
    
    images = {}
    
    #Fusion method
    #--------------------------------------------------------------------------
    classes = classes_list[INDEX_ZERO]
    
    #Get the first train and test list to be the set of train and test
    train_test = list_train_test[INDEX_ZERO]
    for index in range(len(train_test)):
        try:
            train_set = train_test[index][INDEX_TRAIN]
            for img in train_set:
                img_classes = list_images[INDEX_ZERO][img][POS_CLASSES]
                
                if img not in images.keys():
                    images[img] = [img_classes, []]
                images[img][POS_PREDICT].append(zeros(len(list_images[INDEX_ZERO][img][POS_PREDICT])))
        except:
            pass
        
        test_set = train_test[index][INDEX_TEST]
    
        for img in test_set:
            #predict gather the percentages of the predicted class by each of
            # the result in list_images
            predict = []
            for index_images in range(len(list_images)):
                img_predict = list_images[index_images][img][POS_PREDICT]
                predict.append(img_predict)
            mv = majority_voting(predict, index)
            img_classes = list_images[INDEX_ZERO][img][POS_CLASSES]
            
            if img not in images.keys():
                images[img] = [img_classes, []]
            images[img][POS_PREDICT].insert(index, mv)
    #--------------------------------------------------------------------------
    print "Success of the majority voting"
    
    print "Saving the result of the Majority Voting into files"
    save_file(result_path, images, classes, train_test)
    
    return images, classes, train_test
    
def majority_voting(predict_set, index):
    """
    Performs the calculation of the majority voting for the list of results,
    returning a numpy array with the majority class as 1.0, and the other
    classes as 0.0.
    """
    
    size_predict = len(predict_set[INDEX_ZERO][INDEX_ZERO])
    
    list_max_index = []
    mv = zeros(size_predict)
    for predict_list in predict_set:
        predict = predict_list[index]
        pred_max = max(predict)
        index_max = where(array(predict) == pred_max)[INDEX_ZERO][INDEX_ZERO]
        list_max_index.append(index_max)
    
    index_mv = majority(list_max_index)
    mv[index_mv] = 1.0
    
    return mv.tolist()

def majority(list_items):
    """
    Discover the item that is more representative in the list.
    """
    
    counts = {}
    
    for item in list_items:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    
    result = max(counts.iteritems(), key = lambda item: item[1])

    return result[INDEX_ZERO]

def save_file(result_path, images, classes, train_test):
    """
    Save the result of the majority voting into the file.
    """
    
    for index, train_test_set in enumerate(train_test):
        test = train_test_set[INDEX_TEST]
        result_file = open(result_path + str(index) + ".txt", "wb")
        result_file.write(str(classes) + "\n")
        for image_path in test:
            image_classes = images[image_path][POS_CLASSES]
            image_predict = images[image_path][POS_PREDICT][index]
            result_file.write(image_path + " " + str(len(image_classes)) + \
                    " " + str(image_classes) + " " + str(image_predict) + "\n")
        result_file.close()
