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

import os
import sys
from datetime import datetime, timedelta

import config

START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_CLASSES = 0

def main(images, classes, experiment_folder, train_test_method, parameters,
        open_set_experiment, node_id):
    """
    Split the given dataset into train and test set.
     
    Parameters
    ----------
        images : dict, {string : [list, list]}
            A dictionary where each (key, value) pair corresponds to the path
            to an item in the database and a list of structures indexed by the
            item's path, respectively.
            For example, images[path][0] is the list of classes associated with 
            that item. images[path][1] vary thoughout the execution of the
            whole experiment, but is not relevant for this function.
        
        classes : dict, {string : [list of string]}
            A dictionary where each (key, value) pair corresponds to a class in
            the database and a list of paths to the items associated with that
            class, respectively.
            
        experiment_folder : string
            String with the path to the experiment folder, where the files of
            the experiment will be saved.
            
        train_test_method : string
            Name of the specific plugin to be used for the dataset splitting. 
            
        parameters : dict, {string : string}
            Dictionary with the plugin-specific parameters. Usually extracted
            from the experiment's xml.
            
        open_set_experiment : boolean
                Boolean to determinate a open set experiment.
            
        node_id : string
            ID of the train_test_method node in the experiment.
        
    
    Returns
    -------
        new_images : dict, {string : [list, list]}
            A subset of the images parameter containing only the entries 
            generated by the plugin.
            
        new_classes : dict, {string : [list of string]}
            A subset of the classes parameter containing only the entries 
            generated by the plugin.
            
        train_test_list : list of list
            List of the splits generated by the plugin, with each entry
            containing the training and testing sets.
            For example, train_test_list[i][0] is a list of paths of items
            in the training set for the i'th split, while train_test_list[i][1]
            is the testing set for the same split.
            
        train_test_time : float
            Time taken to execute this function.
    """
    
    #CONSTANTS
    POS_CLASSES = 0
    
    print "Train and Test Module"
    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Create new images dictionary with only the images in the train and test
    # set and new classes dictionary
    new_images = {}
    new_classes = {}
    
    #Initiate the calculation of the necessary time to split the dataset
    init_train_test = datetime.now()
    
    #Path to the methods, and import of the plugin
    train_test_path = os.path.join(os.path.dirname(__file__), "..",
                                   "train_test_methods", train_test_method)
    sys.path.append(train_test_path)
    software = __import__("plugin_" + train_test_method)
    train_test_list = software.train_test(images, classes, parameters)
    
    #Open Set
    #--------------------------------------------------------------------------
    if open_set_experiment:
        new_train_test_list = []
    
        #Change the train and test set, moving all images of class None to the
        # test set
        print "Changing the train and test set."
        for train, test in train_test_list:
            new_train = []
            new_test = test
            for img in train:
                if None in images[img][POS_CLASSES]:
                    new_test.append(img)
                else:
                    new_train.append(img)
            new_train_test_list.append([new_train, new_test])
        train_test_list = new_train_test_list
    #--------------------------------------------------------------------------
    
    #Save all train and test files
    total_train_test = len(train_test_list)
    number_train_test = 0
    for i in range(total_train_test):
        print "\tTrain and Test number:", i + 1
        
        train_set = train_test_list[i][0]
        test_set = train_test_list[i][1]
        
        #Paths to save to files containing the paths to the train and test
        train_path = experiment_folder + "iteration:" + str(iteration) + \
                     "-train_test_method-id:" + node_id + "-" + \
                     train_test_method + "-train" + str(i) + ".txt"
        test_path = experiment_folder + "iteration:" + str(iteration) + \
                    "-train_test_method-id:" + node_id + "-" + \
                    train_test_method + "-test" + str(i) + ".txt"
        
        #Save the images path of the train and test
        print "\tSaving file with the train set."
        train_file = open(train_path, "wb")
        for img in train_set:
            line = str(img) + "\n"
            train_file.write(line)
        train_file.close()
        
        number_train_test += 1
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    (number_train_test / (2. * total_train_test))))
        except:
            pass
        
        print "\tSaving file with the test set."
        test_file = open(test_path, "wb")
        for img in test_set:
            line = str(img) + "\n"
            test_file.write(line)
        test_file.close()
        
        number_train_test += 1
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    (number_train_test / (2. * total_train_test))))
        except:
            pass
        
        #Update new images dictionary
        for image in train_set:
            if image not in new_images.keys():
                new_images[image] = images[image]
        for image in test_set:
            if image not in new_images.keys():
                new_images[image] = images[image]
        for image, value in new_images.iteritems():
            classes_list = value[POS_CLASSES]
            for item_class in classes_list:
                if item_class not in new_classes.keys():
                    new_classes[item_class] = []
                if image not in new_classes[item_class]:
                    new_classes[item_class].append(image)
                if not item_class:
                    break
    
    #Calculation of the time to split the dataset
    end_train_test = datetime.now()
    train_test_time = end_train_test - init_train_test
    train_test_time = train_test_time.total_seconds()
    print "Total time to split the dataset:", train_test_time, "seconds."
    
    return new_images, new_classes, train_test_list, train_test_time

def write_tex(train_test_method, train_test_parameters):
    """
    Get the latex-formatted representation of the train/test method.   
    
    Given the parameters, this function simply calls the plugin-specific 
    function that generates the appropriate portion of the final
    latex-formatted document, namely the parameters of the train/test method.
    
    Parameters
    ----------
        train_test_method : string
            Path to the train/test plugin.
        
        train_test_parameters : dict
            Parameters of the train_test_method plugin.
    
    Returns
    -------
        text_result : string
            A string containing the latex-formatted output.
        
    """
    
    print "\tTeX: Train Test"
    
    #Paths
    train_test_method_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                "..", "train_test_methods", train_test_method))
    
    #Import the plugin of the output
    sys.path.append(train_test_method_path)
    software = __import__("plugin_" + train_test_method)
    print software
    
    tex_result = software.write_tex(train_test_method, train_test_parameters)
    
    return tex_result
