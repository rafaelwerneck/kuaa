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

#Python import
import os
import sys
from datetime import datetime, timedelta
from numpy import zeros
import shutil

#Framework import
import config
import util

#CONSTANTS
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_TRAIN = 0
POS_TEST = 1
POS_CLASSES = 0
POS_FV = 1
POS_PREDICT = 1

def main(fv_paths, classes_list, train_test_list, experiment_folder, classifier,
        parameters, descriptor, node_id):
    """
    Main function of the classification module.
    
    This function calls the classify function of the specific classifier and
    calculates the time necessary to its execution.
    
    Parameters
    ----------
        fv_paths : list
            List of files with the feature vectors to the classification.
    
        classes_lis : list
            List of classes of the experiment.
        
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            For example, train_test_list[i][POS_TRAIN] is a list of paths of
            items in the training set for the i'th split, while
            train_test_list[i][POS_TEST] is the testing set for the same split.
            
        
        experiment_folder : string
            String with the path to the experiment folder, where the files of the
            experiment will be saved.
            
        classifier : string
            Name of the classifier plugin.
            
        parameters : dict, {string: string}
            Dictionary with the plugin-specific parameters, extracted from the
            experiment's xml.
            
        descriptor : string
            Name of the descriptor used to extract the features of the
            experiment.
            
        node_id : string
            ID of the classifier plugin node in the experiment.
    
    Returns
    -------
        new_images : dict, {string : [list of string, [list of float]}
            Dictionary containing the list of classes and the list of predicts
            for a given image, indexed by its path.
            
        list_classes : list of list
            List of list of classes for each image for each train test set.
            
        classification_time : float
            Time taken to execute this function.

    """
    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Time calculation of the classification
    init_classification = datetime.now()
    
    print "Classification module"
    
    new_images = {}
    list_classes = []
    
    #Paths
    experiment_path = experiment_folder + "iteration:" + str(iteration) + \
            "-classifier-id:" + node_id + "-" + classifier + "-train_test_"
    experiment_model_path = experiment_folder + "iteration:" + \
            str(iteration) + "-classifier-id:" + node_id + "-" + classifier + \
            "-model_"
    classifiers_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "classifiers", classifier))
    
    #Import the plugin of the classifier
    sys.path.append(classifiers_path)
    software = __import__("plugin_" + classifier)
    print software
    
    #Performs the classification for every test set
    total_classify = len(train_test_list)
    number_classify = 0
    for pos in range(total_classify):
        classification_path = experiment_path + str(pos) + ".txt"
        
        images = util.read_fv_file(fv_paths[pos])
        
        test_imgs, test_class, classification_result, images_classes, \
                model_paths = software.classify(images, classes_list,
                train_test_list[pos][POS_TRAIN],
                train_test_list[pos][POS_TEST], pos, descriptor, parameters)
        print "Success of the classification"
        
        #Move the model of the classifier to the experiment_folder
        if not isinstance(model_paths, list):
            model_paths = [model_paths]
        path_models = os.path.join(experiment_folder, "Models", 'Node ' + \
                str(node_id) + ' - ' + classifier)
        if not os.path.exists(path_models):
            os.makedirs(path_models)
        for item_path in model_paths:
            new_item_path = "{0}_iter_{1}".format(item_path, iteration)
            os.rename(item_path, new_item_path)
            if os.path.exists(os.path.join(path_models, new_item_path.split(os.sep)[-1])):
                os.remove(os.path.join(path_models, new_item_path.split(os.sep)[-1]))
            shutil.move(new_item_path, path_models)

        
        classification_file = open(classification_path, "wb")
        images_classes = list(images_classes)
        classification_file.write(str(images_classes) + '\n')
        for pos_test in range(len(test_imgs)):
            img_path = test_imgs[pos_test]
            img_classes = test_class[pos_test]
            img_predict = classification_result[pos_test]
            
            #Save the new_images with the test set
            np_zeros = zeros(len(images_classes))
            if img_path not in new_images:
                new_images[img_path] = [[img_classes], [np_zeros] * \
                        total_classify]
            new_images[img_path][POS_PREDICT][pos] = img_predict
            
            line = img_path + " " + str(img_classes) + " " + \
                   str(img_predict) + "\n"
            classification_file.write(line)
        classification_file.close()
        
        #Save the classes of this execution
        list_classes.append(images_classes)
        
        #Send to the socket the train and test that was classified
        number_classify += 1
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    (number_classify / total_classify)))
        except:
            pass
    
    #Time calculation of the classification
    end_classification = datetime.now()
    classification_time = end_classification - init_classification
    classification_time = classification_time.total_seconds()
    print "Total classification time:", classification_time, "seconds"
    
    return new_images, list_classes, classification_time
