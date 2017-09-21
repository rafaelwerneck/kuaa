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
import os
from sklearn import preprocessing
from itertools import izip
from subprocess import call
import numpy
import platform

#Framework imports
import grid

#Constants
POS_CLASSES = 0
POS_FV = 1
INDEX_ZERO = 0
HEAD_LINES = 7

def classify(images, classes_list, train_set, test_set, pos_fold, descriptor,
        parameters):
    """
    Performs the classification of the test_set according to the train_set.
    
    Input:
        - images: Python dictionary with all images from the dataset and its
        feature vector.
        - classes: Python dictionary with all classes from the dataset and its
        images.
        - train_set: List of images in the train set.
        - test_set: List of images in the test set.
        - fv_pos: Position of the feature vector according to the train and
        test method.
        - descriptor: Name of the descriptor used in the extraction step, to
        obtain it distance function (not used in this method).
        parameters: Parameters of the libSVM method:
            - Kernel: kernel used in the SVM
            - C: Cost of the SVM
            - degree: degree of the Kernel function
            - gamma: gamma of the Kernel function
            - Cross-Validation: Number of folds in the cross-validation mode
            - Probabilities: Get as predict the probabilities of each class
    Output:
        
    """
    
    print "Classification: libSVM 3.17"
    
    #Get parameters
    libSVM_kernel = kernel(parameters['Kernel'])
    libSVM_c = float(parameters['C'])
    libSVM_degree = int(parameters['degree'])
    libSVM_gamma = float(parameters['gamma'])
    libSVM_cv = int(parameters['Cross-Validation'])
    libSVM_probability = int(parameters['Probabilities'])
    
    #Paths
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    temp_path = os.path.abspath(os.path.join(dirname, "..", "..", "temp"))
    train_path = os.path.join(temp_path, "libSVM.train")
    model_path = os.path.join(temp_path, "libSVM.train.model_" + str(pos_fold))
    test_path = os.path.join(temp_path, "libSVM.test")
    output_path = os.path.join(temp_path, "libSVM.output")

    plugin_train = "svm-train"
    plugin_test = "svm-predict"
    system_platform = [platform.system(), platform.architecture()[0]]
    if system_platform[0] == 'Linux':
        if system_platform[1] == '32bit':
            plugin_train += "_32l"
            plugin_test += "_32l"
    
    #Preprocess each class to a unique value to the classification
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(classes_list)
    print "List of classes of this experiment:", label_encoder.classes_
    
    list_class = []
    list_fv = []
    
    #Read the train file and save the list of class and the list
    #of feature vectors
    for img in train_set:
        list_class.append(images[img][POS_CLASSES][INDEX_ZERO])
        list_fv.append(numpy.array(images[img][POS_FV][INDEX_ZERO]))
    
    list_train = numpy.array(list_fv)
    list_train_class = numpy.array(list_class)
    
    #Given a list of classes, transform each value in this list to a integer
    list_train_class = label_encoder.transform(list_train_class)
    
    #Read the test list and save the list of class and the list
    #of feature vectors
    list_img = test_set
    list_class = []
    list_fv = []
    
    for img in test_set:
        list_class.append(images[img][POS_CLASSES][INDEX_ZERO])
        list_fv.append(numpy.array(images[img][POS_FV][INDEX_ZERO]))
    
    list_test = numpy.array(list_fv)
    list_test_class = numpy.array(list_class)
    
    list_test_class = label_encoder.transform(list_test_class)
    
    #SVM Fit
    #-------------------------------------------------------------------------
    print "\tFit"
    #Create train file
    train_file = open(train_path, "wb")
    for item_class, item_fv in izip(list_train_class, list_train):
        train_file.write(str(item_class) + " ")
        for i in range(len(item_fv)):
            train_file.write(str(i+1) + ":" + str(item_fv[i]) + " ")
        train_file.write("\n")
    train_file.close()

    #Grid-Search
    result, param = grid.find_parameters(train_path, '-gnuplot null')

    libSVM_c = param['c']
    libSVM_gamma = param['g']
    #-----------
    
    cmd_train = """
%s%s.%s%s -t %d -d %d -g %f -c %f -b %d "%s" "%s"
    """ % (dirname, os.sep, os.sep, plugin_train, libSVM_kernel, libSVM_degree, libSVM_gamma,
            libSVM_c, libSVM_probability, train_path, model_path)
    print cmd_train
    
    #Execute SVM-Train
    call(cmd_train, shell = True)
    print "\tEnd Fit"
    #-------------------------------------------------------------------------
    
    #Read configuration of the libSVM model
    model_file = open(model_path, "rb")
    print model_path
    count_head_lines = 0
    for line in model_file.readlines():
        key_model = line.split()[0]
        if key_model == "label":
            model_labels = map(int, line.split()[1:])
            break
    model_file.close()
    model_paths = [model_path]
    
    #SVM Predict
    #-------------------------------------------------------------------------
    #Create test file
    test_file = open(test_path, "wb")
    for item_class, item_fv in izip(list_test_class, list_test):
        test_file.write(str(item_class) + " ")
        for i in range(len(item_fv)):
            test_file.write(str(i+1) + ":" + str(item_fv[i]) + " ")
        test_file.write("\n")
    test_file.close()
    
    #Execute SVM-Predict
    cmd_test = """
%s%s.%s%s -b %d "%s" "%s" "%s"
    """ % (dirname, os.sep, os.sep, plugin_test, libSVM_probability, test_path, model_path,
            output_path)
    print cmd_test
    
    call(cmd_test, shell = True)
    print "\tEnd Predict"
    #-------------------------------------------------------------------------
    
    #Read output file to get the result of the predict
    output_file = open(output_path, "rb")
    if libSVM_probability:
        output_file.readline()
        list_result = []
        for line in output_file.readlines():
            dict_classes = {key_class: 0.0 for key_class in classes_list}
            output_line = map(float, line.split()[1:])
            for class_item, perc in \
                    izip(list(label_encoder.inverse_transform(model_labels)),
                    output_line):
                dict_classes[class_item] = perc
            percentage_list = [dict_classes[key_class] for key_class in \
                    label_encoder.classes_]
            list_result.append(percentage_list)
    else:
        list_predict = map(int, output_file.readlines())
        list_predict = label_encoder.inverse_transform(list_predict)
        
        list_result = []
        for predict in list_predict:
            img_result = [0.0] * len(label_encoder.classes_)
            #Find all predict in the list label_encoder.classes_ and grab the
            #first index
            pos = numpy.where(label_encoder.classes_ == predict)[0][0]
            img_result[pos] = 1.0
            list_result.append(img_result)
    output_file.close()
    
    #Remove temporary files
    os.remove(train_path)
    os.remove(test_path)
    
    return test_set, list_class, list_result, label_encoder.classes_, \
            model_paths

def kernel(kernel_string):
    """
    Transforms the kernel string in an integer to be used by the svm-train.
    """
    
    import sys
    
    if kernel_string == "Linear":
        return 0
    elif kernel_string == "Polynomial":
        return 1
    elif kernel_string == "RBF":
        return 2
    elif kernel_string == "Sigmoid":
        return 3
    else:
        print "Error! Kernel not found!"
        sys.exit(1)
