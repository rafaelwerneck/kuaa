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
from sklearn import metrics
import numpy
from ast import literal_eval
import pylab
from scipy import interp, stats
from math import sqrt

#Framework imports
import util

#CONSTANTS
ZERO = 0

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of the ROC curve.
    
    Compute a binary Receiver operating characteristic (ROC).
    """
    
    #CONSTANTS
    POS_CLASSES = 0
    POS_PREDICT = 1
    
    print "\tOutput: ROC"
    
    #Get parameters
    #Get the positive classes from the parameters
    positive_classes = [item.strip() for item in
            parameters['Positive Label'].split(',')]
    interest_classes = [item_class for item_class in positive_classes if
            item_class in classes_list]
    
    #In case that the interest_classes parameter is empty, consider all classes
    #of the test set
    if interest_classes == [''] or interest_classes == []:
        interest_classes = classes_list
    
    #Output methods
    #-------------------------------------------------------------------------    
    #Create the list with the correct class of each image
    list_class = []
    for img_test in test_set:
        list_class.append(images[img_test][POS_CLASSES][0])
    list_class = numpy.array(list_class)
    
    #Create a dictionary to save the results for each class
    results = {}
    
    #For each class, calculate the ROC curve
    for item_class in interest_classes:
        result_proba = []
        
        #Find the index of the probability of the considered class
        class_index = 0
        for i in range(len(classes_list)):
            if item_class == classes_list[i]:
                class_index = i
                break
        
        #Create the list with the probabilities of each image
        for img_test in test_set:
            result_proba.append(images[img_test][POS_PREDICT][pos_train_test]\
                    [class_index])
        result_proba = numpy.array(result_proba)
        
        #Calculate the False Positive Rates, True Positive Rates and the
        #thresholds used to compute the previous.
        fpr, tpr, thresholds = metrics.roc_curve(list_class,
                result_proba, str(item_class))
        
        results[item_class] = [fpr.tolist(), tpr.tolist(), thresholds.tolist()]

    #-------------------------------------------------------------------------
    print "Success in the calculation of the ROC"
    
    #Delete variables that will not be used
    del interest_classes
    del list_class
    
    return results

def string_file(roc):
    """
    Returns a one line dictionary to be write into the file.
    """
    
    return str(roc)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tROC Curve"
    return "\\subsection{Receiver Operating Characteristic}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates the average ROC from the values in the evaluation_path.
    """
    
    print "\t\tTeX: ROC Curve"
    
    roc_list = []
    tex_string = ""
    
    evaluation_file = open(evaluation_path, "rb")
    lines = evaluation_file.readlines()
    for each_line in lines:
        roc_list.append(literal_eval(each_line))
    evaluation_file.close()
    
    if not isinstance(roc_list[ZERO], list):
        roc_list = [[roc_item] for roc_item in roc_list]
    
    for index_roc, roc_item in enumerate(roc_list):
        for key_class in classes:
            flag_class = False # Flag to avoid classes outside of the result of ROC
            mean_tpr = 0.0
            mean_fpr = numpy.linspace(0, 1, 100)
            all_tpr = []
            
            for index, value in enumerate(roc_item):
                if key_class not in value.keys():
                    flag_class = True
                    break
                fpr, tpr, thresholds = value[key_class]
                all_tpr.append(interp(mean_fpr, fpr, tpr))
            
            # In case Flag was activate, this class is not in the ROC result
            if flag_class:
                continue
            
            pylab.clf()
            mean_tpr = numpy.mean(all_tpr, axis=0)
            std_tpr = numpy.std(all_tpr, axis=0)
            conf_tpr = [stats.t.interval(0.95, len(all_tpr) - 1,
                    loc = mean_tpr[index],
                    scale = std_tpr[index] / sqrt(len(all_tpr)))
                    for index in range(len(mean_tpr))]
            mean_auc = metrics.auc(mean_fpr, mean_tpr)
            pylab.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' \
                    % mean_auc, lw=2)
            pylab.plot(mean_fpr, conf_tpr, ':', label="Confidence Interval (95%)")
            
            pylab.xlim([-0.05, 1.05])
            pylab.ylim([-0.05, 1.05])
            pylab.xlabel('False Positive Rate')
            pylab.ylabel('True Positive Rate')
            pylab.title('Receiver operating characteristic, class %s' % key_class)
            pylab.legend(loc="lower right")
            file_fig = os.path.join("experiments", "Experiment_" + experiment_id,
                                str(node_id) + "_roc_" + str(key_class) + "_fold_" + str(index_roc))
            pylab.savefig(file_fig + ".pdf")
            
            tex_string += """
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\columnwidth]{"%s"}
    \\caption{Receiver Operating Characteristic of class %s of Node %s of Fold %s.}
    \\label{fig:cm_%s_%s}
\\end{figure}
            """ % (os.path.abspath(file_fig), str(key_class), node_id,
                    str(key_class), node_id, index_roc)

    return tex_string
