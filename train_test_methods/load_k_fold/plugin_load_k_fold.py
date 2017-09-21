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

# Python imports
import os

def train_test(images, classes, parameters):
    """
    Train test method that read files containing a training set and testing set
    of a K-Fold.

    Input:
        images: python dictionary containing as keys all images of the dataset.
        classes: python list containing all classes of the dataset.
        parameters: parameters of the train test method:
            - Number of Folds: integer with the number of folds
            - Training Fold $i: path to the file containing the training set
                number $i.
            - Testing Fold $i: path to the file containing the testing set 
                number $i.
    Output:
        list_train_test: python list containing the images from each train and
            test.
    """
    
    print "\tLoad K-Folds"
    
    # Get parameters
    number_of_folds = parameters['Number of Folds']
    train_path = []
    test_path = []
    for number_fold in range(number_of_folds):
        train_path.append(parameters["Training Fold {0}".format(number_fold)])
        test_path.append(parameters["Testing Fold {0}".format(number_fold)])
    
    list_train_test = []
    
    for number_fold in range(number_of_folds):
        print "\t\tReading fold: {0}".format(number_fold)
        
        # Read training file
        train = []
        train_file = open(train_path[number_fold], "rb")
        for line in train_file.readlines():
            
            train.append(line[:-1])
        train_file.close()
        
        # Read testing file
        test = []
        test_file = open(test_path[number_fold], "rb")
        for line in test_file.readlines():
            test.append(line[:-1])
        test_file.close()
        
        list_train_test.append([train, test])
    
    return list_train_test

def write_tex(method, parameters):
    """
    Function to write into the tex file the parameters of the Load K-Fold
    method.
    """
    
    print "\t\tTeX: Load K-Fold"
    
    # Get parameters
    param_folds = parameters['Number of Folds']
    
    tex_string = """
\\section{Protocol: Load K-Fold}
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{cc}
        Parameter & Value \\\\
        \\hline
        Number of Folds & %d \\\\
    \\end{tabular}
    \\caption{Parameters of the Load K-Fold Method.}
    \\label{tab:%s_param_%d}
\\end{table}
    """ % (param_folds, method, param_folds)
    
    return tex_string
