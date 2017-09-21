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
from sklearn.cross_validation import KFold

def train_test(images, classes, parameters):
    """
    Divides the dictionary keys of the images in folds, according to the
    parameters:
        - Number of Folds: number of folds that will be created.
    
    The first n % n_folds folds have size n // n_folds + 1, other folds have
    size n // n_folds.
    """
    
    print "Train and Test: K-Fold"
    
    print parameters
    
    #Get parameters
    param_folds = parameters['Number of Folds']
    
    list_train_test = []
    
    #Split the dataset into train and test
    print "\tSpliting the dataset into train and test."
    
    #Train and Test Split
    #--------------------------------------------------------------------------
    len_images = len(images)
    images_keys = images.keys()
    k_fold = KFold(len_images, param_folds)
    
    #Transform the index of the KFold function into the keys of the images
    #dictionary
    for train_index, test_index in k_fold:
        train = []
        test = []
        for index in train_index:
            train.append(images_keys[index])
        for index in test_index:
            test.append(images_keys[index])
        list_train_test.append([train, test])
    #--------------------------------------------------------------------------
    
    return list_train_test

def write_tex(method, parameters):
    """
    Write the TeX section about the train and test method of the experiment.
    """
    
    print "\t\tTeX: K-Fold"
    
    # Get parameters
    param_folds = parameters['Number of Folds']
    
    tex_string = """
\\section{Protocol: K-Fold}
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{cc}
        Parameter & Value \\\\
        \\hline
        Number of Folds & %d \\\\
    \\end{tabular}
    \\caption{Parameters of the K-Fold Method.}
    \\label{tab:%s_param_%d}
\\end{table}
    """ % (param_folds, method, param_folds)
    
    return tex_string
