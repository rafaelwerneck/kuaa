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
from ast import literal_eval
from numpy import array

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of False Negatives.
    
    Calculates, for each class in the parameters, the number of items that are
    from this class, and was wrong classified.
    """
    
    #CONSTANTS
    POS_CLASSES = 0
    POS_PREDICT = 1
    
    print "\tOutput: FALSE NEGATIVE"
    
    #Get parameters
    #Get the positive classes from the parameters
    positive_classes = [item.strip() for item in
            parameters['Classes'].split(',')]
    interest_classes = [item_class for item_class in positive_classes if
            item_class in classes_list]
    
    #In case that the interest_classes parameter is empty, consider all classes
    #of the test set
    if interest_classes == [''] or interest_classes == []:
        interest_classes = classes_list
    
    #Output methods
    #-------------------------------------------------------------------------
    #Create a dictionary to save the results for each class
    results = {}
    
    #For each class, calculate the false negative
    for item_class in interest_classes:
    
        false_negative = 0
        all_images = len(test_set)
    
        #Trasverse the test set to get the path of the especific images
        for img_test in test_set:
            actual_class = images[img_test][POS_CLASSES][0]
            if  actual_class != item_class:
                continue
            predict_proba = images[img_test][POS_PREDICT][pos_train_test]
            img_predict_index = 0
            for i in range(len(classes_list)):
                if predict_proba[i] > predict_proba[img_predict_index]:
                    img_predict_index = i
            predict_class = classes_list[img_predict_index]
            
            if item_class != predict_class:
                false_negative += 1
        
        results[item_class] = [false_negative, false_negative / all_images]
    
    #-------------------------------------------------------------------------
    print "End of the False Negative method."
    
    #Clean the variables used in the method
    del interest_classes
    
    return results

def string_file(false_negative):
    """
    Returns a one line dictionary with the false negative to be write into the
    file.
    """
    
    return str(false_negative)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tFalse Negative"
    return "\\subsection{False Negative}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates an average number of false negatives in the experiment.
    """
    
    #Constants
    INDEX_FN = 0
    INDEX_PERC = 1
    INDEX_ZERO = 0
    
    print "\t\tTeX: False Negative"
    
    fn_dicts = []
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        fn_dicts.append(literal_eval(line))
    evaluation_file.close()
    
    avg_fn = {}
    for class_key in fn_dicts[INDEX_ZERO].iterkeys():
        fn_list = []
        perc_list = []
        for fn in fn_dicts:
            fn_list.append(fn[class_key][INDEX_FN])
            perc_list.append(fn[class_key][INDEX_PERC])
        avg_fn[class_key] = [array(fn_list).mean(), array(perc_list).mean()]
    
    evaluation_file = open(evaluation_path, "ab")
    evaluation_file.write("\nAverage False Negative\n")
    evaluation_file.write(str(avg_fn))
    evaluation_file.close()
    
    tex_string = """
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{ccc}
        & Number of False Negatives & Percentage of Total (\\%) \\\\
        \\hline"""
    
    for class_key in avg_fn.iterkeys():
        tex_string += """
        %s & %.2f & %.2f \\\\""" % (class_key, avg_fn[class_key][INDEX_FN],
                avg_fn[class_key][INDEX_PERC] * 100)
                
    tex_string += """
    \\end{tabular}
    \\caption{Mean Number of False Negatives and Percentage of False Negatives of Node %s}
    \\label{tab:fn_%s}
\\end{table}
    """ % (node_id, node_id)
    
    return tex_string
