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
import numpy
from itertools import izip
from ast import literal_eval

#Constants
INDEX_ZERO = 0
POS_CLASSES = 0
POS_PREDICT = 1

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of the normalized accuracy score.

    Calculates the accuracy score using the entries of test set and save
    the result in a string to be returned by the function.
    """
    
    print "\tOutput: NORMALIZED ACCURACY SCORE"

    # Get parameters

    # Evaluation method
    #-------------------------------------------------------------------------
    list_class = []
    result_class = []

    for img_test in test_set:
        list_class.append(images[img_test][POS_CLASSES][INDEX_ZERO])
        classes_proba = images[img_test][POS_PREDICT][pos_train_test]
        img_predict_index = 0
        for i in range(len(classes_list)):
            if classes_proba[i] > classes_proba[img_predict_index]:
                img_predict_index = i
        result_class.append(classes_list[img_predict_index])

    # Calculates the accuracy score
    known_data = [(predict, test) for predict, test in izip(result_class, list_class) if test != None]
    unknown_data = [(predict, test) for predict, test in izip(result_class, list_class) if test == None]
    
    known_accuracy = sum([predict == test for predict, test in known_data]) / float(len(known_data))
    unknown_accuracy = sum([predict == test for predict, test in unknown_data]) / float(len(unknown_data)) if len(unknown_data) > 0 else 1.0
    
    accuracy_per_class = [numpy.mean([predict == test for predict, test in izip(result_class, list_class) if test == item_class]) for item_class in classes_list]
    global_na = float(numpy.mean(accuracy_per_class))
    
    known_classes = classes_list[:]
    known_classes.remove(None)
    naks = [numpy.mean([predict == test for predict, test in izip(result_class, list_class) if test == item_class]) for item_class in known_classes]
    naks = float(numpy.mean(naks))
    
    returns = {
            'Accuracy of Known Samples': [known_accuracy],
            'Accuracy of Unknown Samples': [unknown_accuracy],
            'Normalized Accuracy': [(known_accuracy + unknown_accuracy) / 2],
            'Global Normalized Accuracy': [global_na],
            'Normalized Accuracy of Known Samples': [naks]
            }
    #-------------------------------------------------------------------------
    print "\tSuccess in the calculation of the accuracy score"

    # Delete variables not used
    del list_class
    del result_class

    return returns

def string_file(normalized_accuracy):
    """
    Returns a one line normalized accuracy to be write into the file.
    """
    
    return str(normalized_accuracy)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tNormalized Accuracy Score"
    return "\\subsection{Normalized Accuracy Score}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates the average normalized accuracy score from the accuracy scores in the
    evaluation_path.
    """
    
    from numpy import array
    from scipy import stats
    from math import sqrt
    
    print "\t\tTeX: Normalized Accuracy Score"
    
    na_dicts = []
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        na_dicts.append(literal_eval(line))
    evaluation_file.close()
    
    avg_na = {}
    for evaluation in na_dicts[INDEX_ZERO].iterkeys():
        na_list = []
        for na in na_dicts:
            na_list.append(na[evaluation][INDEX_ZERO])
        avg_na[evaluation] = numpy.array(na_list).mean()
    
    evaluation_file = open(evaluation_path, "ab")
    for evaluation, item_na in avg_na.iteritems():
        evaluation_file.write("\n" + str(evaluation) + "\n")
        evaluation_file.write(str(item_na))
    evaluation_file.close()
    
    tex_string = """
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{cc}
        & Accuracy \\\\
        \\hline"""
    
    for evaluation, value in avg_na.iteritems():
        tex_string += """
        %s & %.2f \\\\""" % (evaluation, value * 100)
    
    tex_string += """
    \\end{tabular}
    \\caption{Normalized Accuracy Score of Node %s}
    \\label{tab:acc_%s}
\\end{table}
    """ % (node_id, node_id)
    
    return tex_string
