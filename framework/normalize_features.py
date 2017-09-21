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
from datetime import datetime, timedelta
import sys
import multiprocessing

import util

import config
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_CLASSES = 0
POS_FV = 1
POS_TRAIN = 0
POS_TEST = 1

#Global variables
str_buffer = ""
normalized_path = ""
train_param = {}
number_images = 0
node_id = 0
total_images = 0
curr_progress = -1

def main(images, train_test_list, experiment_folder, normalizer, parameters,
         id_node):
    """
    Call the descriptor plugin to extract the descriptors of the given images.
    
    Parameters
    ----------
        images : dict, {string : [list, list]}
            The keys of the dictionary are the paths to the images whose
            descriptors will be extracted.
            
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            For example, train_test_list[i][0] is a list of paths of items
            in the training set for the i'th split, while train_test_list[i][1]
            is the testing set for the same split.
            The normalization will be done across the images in each 
            training set.
        
        experiment_folder : string
            Path to the experiment's folder.
                    
        normalizer : string
            Name of the normalization mehthod, which will be used to locate 
            the plugin.
            
        parameters : dict, {string : string}
            Dictionary with the plugin-specific parameters. Usually extracted 
            from the experiment's xml.
            
        id_node : string
            ID of the normalization plugin node in the experiment.   
    
    Returns
    -------
        norm_fv_paths : list
            List with the paths of the files with the result of the
            normalization.
            
        normalize_time : float
            Time taken to execute this function.
    
    """
    
    global str_buffer
    global normalized_path
    global train_param
    global node_id
    global total_images
    global number_images
    
    node_id = id_node
    number_images = 0
    total_images = 0
    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    # The i'th entry store the path to the file with the fv's of the i'th
    # train/test split.
    norm_fv_paths = []
    
    print "Normalization Module"
    
    #Calculate the normalization time
    init_normalize = datetime.now()
    
    #Import
    path_normalizers = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "normalizers", normalizer))
    sys.path.append(path_normalizers)
    software = __import__("plugin_" + normalizer)
    print software
    
    #Performs the normalization for every train and test
    total_normalize = len(train_test_list)
    for train, test in train_test_list:
        total_images += len(train) + len(test)
    for i in range(total_normalize):
        #Reset the parameters of the training step
        train_param = {}
        
        #Paths
        normalized_path = experiment_folder + "iteration:" + str(iteration) + \
                          "-normalizer-id:" + node_id + "-" + normalizer + \
                          "-train_test" + str(i) + ".txt"
        norm_fv_paths.append(normalized_path)
        
        num_cores = int(multiprocessing.cpu_count())
        print "Number of cores used: ", num_cores
        
        print "Normalizing the training set"
        pool = multiprocessing.Pool(num_cores)
        #Normalization of the train set
        for img in train_test_list[i][POS_TRAIN]:
            img_classes = images[img][POS_CLASSES]
            img_fv = images[img][POS_FV]
            pool.apply_async(software.normalize, args = (img, images,
                    train_test_list[i][POS_TRAIN], i, parameters, "train",
                    train_param), callback = save_norm)
        pool.close()
        pool.join()
        
        print "Normalizing the testing set"
        pool = multiprocessing.Pool(num_cores)
        #Normalization of the test set
        for img in train_test_list[i][POS_TEST]:
            img_classes = images[img][POS_CLASSES]
            img_fv = images[img][POS_FV]
            pool.apply_async(software.normalize, args = (img, images,
                    train_test_list[i][POS_TEST], i, parameters, "test",
                    train_param), callback = save_norm)
        pool.close()
        pool.join()
        
        print "Saving rest of the buffer"
        normalized_file = open(normalized_path, "ab")
        normalized_file.write(str_buffer)
        normalized_file.close()
        
        #Clean buffer
        str_buffer = ""
    
    #End of the normalization
    print "Success of the normalization"
    
    end_normalize = datetime.now()
    
    #Calculate the total normalization time
    normalize_time = end_normalize - init_normalize
    normalize_time = normalize_time.total_seconds()
    print "Total normalize time: ", normalize_time, " seconds"
    
    return norm_fv_paths, normalize_time

def save_norm(result):
    """
    Buffer a normalized fv and save to disk when the buffer is full.    
    
    Callback of the Pool.apply_async, receive the result of the operation made 
    by the software.extract and add it in the buffer. In case the buffer exceed
    the BUFFER_LIMIT, put it in the normalized_file and clean the buffer.
    
    Parameters
    ----------
        result : list, [img_path, len(img_classes), img_classes, norm_fv, train_param]
            Used to save a feature vector to disk in the framework's format.
    
    Returns
    -------
        None
        
    """
    
    global str_buffer
    global normalized_path
    global train_param
    global number_images
    global node_id
    global total_images
    global curr_progress

    #Constants
    BUFFER_LIMIT = int(1E4)
    
    print "Get result from process"
    new_image_path = result[0]
    new_number_classes = result[1]
    new_image_classes = result[2]
    new_image_fv = result[3]
    new_train_param = result[4]
    str_buffer = str_buffer + new_image_path + " " + str(new_number_classes) + \
                 " " + str(new_image_classes) + " " + str(new_image_fv)
    
    #Updates the parameters of the train step
    train_param.update(new_train_param)
    
    #In case that the buffer is greater than BUFFER_LIMIT, save the buffer
    #in the normalized_path
    if sys.getsizeof(str_buffer) >= BUFFER_LIMIT:
        print "Saving the buffer..."
        normalized_file = open(normalized_path, "ab")
        normalized_file.write(str_buffer)
        normalized_file.close()
        str_buffer = ""
    str_buffer = str_buffer + "\n"
    
    number_images += 1
    progress = number_images / total_images
    
    if int(100 * progress) > curr_progress:
        curr_progress = int(100 * progress)
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    progress))
        except:
            pass
