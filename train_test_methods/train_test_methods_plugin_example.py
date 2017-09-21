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

def train_test(images, classes, parameters):
    """
    Divides the dictionary keys of the images in train and test set.
    
    input:
        images: dict, {string : [list, list]}
            A dictionary where each (key, value) pair corresponds to the path
            to an item in the database and a list of structures indexed by the
            item's path, respectively.
            For example, images[path][0] is the list of classes associated with 
            that item. images[path][1] vary thoughout the execution of the
            whole experiment, but is not relevant for this function.
    
    output:
        list_train_test: list containing the list of train images and the list
            of test images.
    """
    
    print "Train and Test: EXAMPLE"
    
    # Get parameters
    
    list_train_test = []
    
    # Split the dataset into train and test
    print "\tSpliting the dataset into train and test."
    
    # Train and Test Split
    #--------------------------------------------------------------------------
    # Create custom method
    #--------------------------------------------------------------------------
    
    return list_train_test

def write_tex(method, parameters):
    """
    Write the TeX section about the train and test method of the experiment.
    """
    
    print "\t\tTeX: EXAMPLE"
    
    # Get parameters of the train and test method.
    
    tex_string = ""
    
    return tex_string
