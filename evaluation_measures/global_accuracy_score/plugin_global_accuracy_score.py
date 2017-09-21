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
    
from sklearn import metrics
from itertools import izip
import numpy

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of the global accuracy score.

    Calculates the accuracy score using the entries of test set and save
    the result in a string to be returned by the function.
    """

    # CONSTANTS
    POS_CLASSES = 0
    POS_PREDICT = 1

    print "\tOutput: GLOBAL ACCURACY SCORE"

    # Get parameters

    # Output methods
    #-------------------------------------------------------------------------
    list_class = []
    result_class = []

    for img_test in test_set:
        list_class.append(str(images[img_test][POS_CLASSES][0]))
        classes_proba = images[img_test][POS_PREDICT][pos_train_test]
        img_predict_index = 0
        for i in range(len(classes_list)):
            if classes_proba[i] > classes_proba[img_predict_index]:
                img_predict_index = i
        result_class.append(str(classes_list[img_predict_index]))

    # Calculates the accuracy score
    accuracy = metrics.accuracy_score(list_class, result_class)
    
    # Calculates the normalized global accuracy score
    accuracy_per_class = []
    for item_class in classes_list:
        accuracy_class = [predict == test for predict, test in izip(result_class, list_class) if test == item_class]
        if len(accuracy_class):
            accuracy_per_class.append(numpy.mean(accuracy_class))
    global_na = float(numpy.mean(accuracy_per_class))
    #-------------------------------------------------------------------------
    print "\tSuccess in the calculation of the global accuracy score"

    # Delete variables not used
    del list_class
    del result_class

    return {"Global Accuracy": accuracy, "Normalized Global Accuracy": global_na}

def string_file(accuracy):
    """
    Returns a one line accuracy to be write into the file.
    """
    
    return str(accuracy)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tGlobal Accuracy Score"
    return "\\subsection{Global Accuracy Score}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates the average global accuracy score from the accuracy scores in the
    evaluation_path.
    """
    
    from scipy import stats
    from math import sqrt
    from ast import literal_eval
    
    ZERO = 0
    
    print "\t\tTeX: Global Accuracy Score"
    
    global_acc_list = []
    tex_string = ""
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        global_acc_list.append(literal_eval(line))
    evaluation_file.close()
    
    for key in global_acc_list[ZERO].iterkeys():
        acc_list = []
        for global_acc in global_acc_list:
             acc_list.append(float(global_acc[key]))
        
        avg_acc = numpy.array(acc_list).mean()
        std_acc = numpy.array(acc_list).std()
        interval = stats.t.interval(0.95, len(acc_list) - 1, loc = avg_acc,
                scale = std_acc / sqrt(len(acc_list)))
        conf = avg_acc - interval[0]
        
        evaluation_file = open(evaluation_path, "ab")
        evaluation_file.write("\n{0}\nAverage Global Accuracy Score\n".format(key))
        evaluation_file.write(str(avg_acc))
        evaluation_file.write("\nStandard Deviation\n")
        evaluation_file.write(str(std_acc))
        evaluation_file.write("\nConfidence Interval (95%)\n")
        evaluation_file.write(str(conf))
        evaluation_file.close()
        
        tex_string += """
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{ccc}
        Mean & Deviation & Confidence Interval (95\\%%)\\\\
        \\hline
        %.2f & %.2f & %.2f
    \\end{tabular}
    \\caption{Average, Standard Deviation and Confidence Interval of the %s of Node %s}
    \\label{tab:acc_%s}
\\end{table}
        """ % (avg_acc * 100, std_acc * 100, conf * 100, key, node_id, node_id)
    
    return tex_string
