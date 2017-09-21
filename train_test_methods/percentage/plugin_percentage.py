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
from sklearn.cross_validation import train_test_split

#Constants
ONE_HALF = 0.5

def train_test(images, classes, parameters):
    """
    Divides the dictionary keys of the images in train and test set, according
    to the parameters:
        - Select per Classes: The division in the train and test set will be
            considering each class of the dataset;
        - Train Size: Value of the division of the set. Select the percentage
            of the database that will be part of the train set;
        - Test Size: Value of the division of the set. Select the percentage
            of the database that will be part of the test set;
            
    Later, this function save the train and test files in the result_path.
    """
    
    print "Train and Test: Percentage"
    
    #Get parameters
    param_classes = parameters['Select per Classes']
    param_train_size = parameters['Train Size']
    param_test_size = parameters['Test Size']
    
    list_train_test = []
    
    #Split the dataset into train and test
    print "\tSpliting the dataset into train and test."
    
    #Train and Test Split
    #--------------------------------------------------------------------------
    img_train = []
    img_test = []
    
    #Considering each class
    if param_classes:
        for id_class in classes.iterkeys():
            print "\t\tClass", id_class
            try:
                if param_train_size == -1:
                    class_train, class_test = train_test_split(
                            classes[id_class], test_size=param_test_size)
                elif param_test_size == -1:
                    class_train, class_test = train_test_split(
                            classes[id_class], train_size=param_train_size)
                else:
                    class_train, class_test = train_test_split(
                            classes[id_class], train_size=param_train_size,
                            test_size=param_test_size)
            except:
                if param_train_size >= 1:
                    class_train = classes[id_class]
                    class_test = []
                elif param_test_size >= 1:
                    class_train = []
                    class_test = classes[id_class]
                else:
                    #In case that the sum of the percentages in the train set
                    # and the test set be over one hundred percente, half
                    # of the images will be in the train set and the other half
                    # in the test set
                    class_train, class_test = \
                            train_test_split(classes[id_class],
                            train_size=ONE_HALF, test_size=ONE_HALF)
            img_train.extend(class_train)
            img_test.extend(class_test)
    #Considering the dataset as whole
    else:
        try:
            if param_train_size == -1:
                class_train, class_test = train_test_split(
                        images.keys(), test_size=param_test_size)
            elif param_test_size == -1:
                class_train, class_test = train_test_split(
                        images.keys(), train_size=param_train_size)
            else:
                class_train, class_test = train_test_split(
                        images.keys(), train_size=param_train_size,
                        test_size=param_test_size)
        except:
            if param_train_size >= 1:
                class_train = images.keys()
                class_test = []
            elif param_test_size >= 1:
                class_train = []
                class_test = images.keys()
            else:
                #In case that the sum of the percentages in the train set and
                # the test set be over one hundred percente, half
                # of the images will be in the train set and the other half
                # in the test set
                class_train, class_test = train_test_split(images.keys(),
                        train_size=ONE_HALF, test_size=ONE_HALF)
        img_train.extend(class_train)
        img_test.extend(class_test)
    
    list_train_test.append([img_train, img_test])
    #--------------------------------------------------------------------------
    
    return list_train_test

def write_tex(method, parameters):
    """
    Write the TeX string with the parameters of the train and test method.
    """
    
    print "\t\tTeX: Percentage"
    
    # Get parameters
    param_classes = str(parameters['Select per Classes'])
    param_train_size = parameters['Train Size']
    param_test_size = parameters['Test Size']
    
    tex_string = """
\\section{Protocol: Percentage}
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{cc}
        Parameter & Value \\\\
        \\hline
        Select per Classes & %s \\\\
        Train Size & %.2f \\\\
        Test Size & %.2f \\\\
    \\end{tabular}
    \\caption{Parameters of the Percentage Method.}
    \\label{tab:%s_param_%s_%.2f_%.2f}
\\end{table}
    """ % (param_classes, param_train_size, param_test_size, method,
            param_classes, param_train_size, param_test_size)
    
    return tex_string
