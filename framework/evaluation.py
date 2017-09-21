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

#Python imports
import os
import sys
from datetime import datetime, timedelta
import socket
import itertools
from numpy import ndarray

#Framework imports
import config

#CONSTANTS
POS_TESTS = 1
INDEX_ZERO = 0
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_ZERO = 0

def main(images, train_test_list, classes_list, experiment_folder, evaluation_measure, parameters, node_id):
    """
    Main function of the output module.
    
    This function calls the plugin of the output desired and calculates the
    time of the execution.
    
    Parameters
    ----------
        images : dict, {string : [list of string, [list of float]}
            Dictionary containing the list of classes and the list of predicts
            for a given image, indexed by its path.
            
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            For example, train_test_list[i][POS_TRAIN] is a list of paths of
            items in the training set for the i'th split, while
            train_test_list[i][POS_TEST] is the testing set for the same split.
            
        classes_list : list of list
            List of list of classes for each image for each train test set.
            
        experiment_folder : string
            String with the path to the experiment folder, where the files of the
            experiment will be saved.
            
        evaluation_measure : string
            Name of the evaluation measure plugin.
        
        parameters : dict, {string: string}
            Dictionary with the plugin-specific parameters, extracted from the
            experiment's xml.
            
        node_id : string
            ID of the evaluation plugin node in the experiment.
    
    Returns
    -------
        evaluation_time : float
            Time taken to execute this function.
            
        evaluation_path : string
            Path to the file with the result of the evaluation measure.
    
    """
    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Time calculation of the output
    init_evaluation = datetime.now()
    
    print "Evaluation Measures"
    
    #Paths
    evaluation_methods_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "..", "evaluation_measures", evaluation_measure))
    
    #Import the plugin of the output
    sys.path.append(evaluation_methods_path)
    software = __import__("plugin_" + evaluation_measure)
    print software
    
    #List of outputs
    evaluation_list = []
    
    #Output for every classification made
    total_evaluation = len(train_test_list)
    number_evaluation = 0
    for index in range(total_evaluation):
        evaluation_path = experiment_folder + "evaluation_method-id:" + node_id + "-" + \
                      evaluation_measure + ".txt"
        
        #Call the output method from the software plugin
        #The output parameter is temporary, until decide if the result of the
        #output method is better saved in the plugin, or in the module
        result = software.evaluation(images, train_test_list[index][POS_TESTS],
                                 classes_list[index], index, parameters)
        print "Success of the output method"
        
        #Output list append the result from each fold
        #   [R1, R2, R3]
        evaluation_list.append(result)
        
        #Send to the socket the train and test that has the output calculated
        number_evaluation += 1
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    (number_evaluation / total_evaluation)))
        except:
            pass
            
    evaluation_file = open(evaluation_path, "a")
    for evaluation in evaluation_list:
        evaluation_file.write(software.string_file(evaluation) + "\n")
    evaluation_file.close()
    
    #Time calculation of the output
    end_evaluation = datetime.now()
    evaluation_time = end_evaluation - init_evaluation
    evaluation_time = evaluation_time.total_seconds()
    print "Total output time:", evaluation_time, "seconds"
    
    return evaluation_time, evaluation_path

def write_tex(evaluation_method, evaluation_path_id):
    """
    Function to write the tex file considering all iterations of the
    experiment.
    
    Parameters
    ----------
        evaluation_method : string
            Name of the evaluation measure plugin.
            
        evaluation_path_id : string
            ID of the evaluation plugin node in the experiment.
    
    Returns
    -------
        tex_result : string
            A string containing the latex-formatted evaluation measure.
    """
    
    print "\tTeX: Output"
    
    #Paths
    evaluation_methods_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               "..", "outputs", evaluation_method))
    
    #Import the plugin of the output
    sys.path.append(evaluation_methods_path)
    software = __import__("plugin_" + evaluation_method)
    print software
    
    tex_result = software.tex_name()
    
    for evaluation_path, classes_list, node_id in evaluation_path_id:
        tex_result += software.write_tex(evaluation_path, classes_list, node_id)
    
    return tex_result
