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

import os
from sklearn import metrics
import numpy
from ast import literal_eval
    
#CONSTANTS
POS_CLASSES = 0
POS_PREDICT = 1
INDEX_ZERO = 0

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of the F-Measure.
    """
    
    print "\tOutput: F-Measure"
    
    #Get parameters
    
    #Evaluation method
    #--------------------------------------------------------------------------
    list_class = []
    result_class = []
    
    for img_test in test_set:
        list_class.append(images[img_test][POS_CLASSES][INDEX_ZERO])
        classes_proba = images[img_test][POS_PREDICT][pos_train_test]
        img_predict_index = INDEX_ZERO
        for i in range(len(classes_list)):
            if classes_proba[i] > classes_proba[img_predict_index]:
                img_predict_index = i
        print "\t\t", classes_proba.index(max(classes_proba)), img_predict_index
        result_class.append(classes_list[img_predict_index])
            
    #Calculate the confusion matrix
    print "result_class:", result_class, "type:", type(result_class)
    print "list_class:", list_class, "type:", type(list_class)
    cm = metrics.confusion_matrix(numpy.array(list_class),
                                  numpy.array(result_class))
    
    def precisionMacro(indexes):
        def f(i):
            TP = cm[i, i]
            FP = sum(cm[:, i]) - TP
            val = TP / (TP + FP) if TP != 0.0 else 0.0
            return val
        precision = numpy.mean(map(f, indexes))
        return precision

    def recallMacro(indexes):
        def f(i):
            TP = cm[i, i]
            FN = sum(cm[i, :]) - TP
            val = TP / (TP + FN) if TP != 0.0 else 0.0
            return val
        recall = numpy.mean(map(f, indexes))
        return recall

    def precisionMicro(indexes):
        def f1(i):
            TP = cm[i, i]
            return TP

        def f2(i):
            TP = cm[i, i]
            FP = sum(cm[:, i]) - TP
            return TP + FP

        num, den = sum(map(f1, indexes)), sum(map(f2, indexes))
        precision = num / den if den != 0 else 0.0
        return precision

    def recallMicro(indexes):
        def f1(i):
            TP = cm[i, i]
            return TP

        def f2(i):
            TP = cm[i, i]
            FN = sum(cm[i, :]) - TP
            return TP + FN

        num, den = sum(map(f1, indexes)), sum(map(f2, indexes))
        recall = num / den if den != 0 else 0.0
        return recall
    
    indexes = range(len(classes_list) - 1)
    precision_macro = precisionMacro(indexes)
    recall_macro = recallMacro(indexes)
    precision_micro = precisionMicro(indexes)
    recall_micro = recallMicro(indexes)
    
    fmeasure_macro = ((2.0 * precision_macro * recall_macro) / (precision_macro + recall_macro)
                      if precision_macro != 0.0 or recall_macro != 0.0
                      else 0.0)
    fmeasure_micro = ((2.0 * precision_micro * recall_micro) / (precision_micro + recall_micro)
                      if precision_micro != 0.0 or recall_micro != 0.0
                      else 0.0)
    
    #Create a dictionary to save the results for each f-measure
    results = {
            'F-Measure Macro': [fmeasure_macro],
            'F-Measure Micro': [fmeasure_micro]
            }
    #--------------------------------------------------------------------------
    print "Success in the calculation of the F-Measure"
    
    #Delete variables that will not be used
    del list_class
    del result_class
    
    return results

def string_file(f_measure):
    """
    Returns a one line dictionary with the f-measure to be write into the
    file.
    """
    
    return str(f_measure)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tF-Measure"
    return "\\subsection{F-Measure}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates an average number of f-measure in the experiment.
    """
    
    #Constants
    INDEX_ZERO = 0
    
    print "\t\tTeX: F-Measure"
    
    f_measure_dicts = []
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        f_measure_dicts.append(literal_eval(line))
    evaluation_file.close()
    
    avg_f_measure = {}
    for evaluation in f_measure_dicts[INDEX_ZERO].iterkeys():
        f_measure_list = []
        for f_measure in f_measure_dicts:
            f_measure_list.append(f_measure[evaluation][INDEX_ZERO])
        avg_f_measure[evaluation] = numpy.array(f_measure_list).mean()
    
    evaluation_file = open(evaluation_path, "ab")
    evaluation_file.write("\nAverage F-Measure\n")
    evaluation_file.write(str(avg_f_measure))
    evaluation_file.close()
    
    tex_string = """
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{cc}
        & Mean of F-Measure \\\\
        \\hline"""
    
    for evaluation, f_measure in avg_f_measure.iteritems():
        tex_string += """
        %s & %.2f \\\\""" % (evaluation, f_measure * 100)
                
    tex_string += """
    \\end{tabular}
    \\caption{Mean F-Measure Evaluation of Node %s}
    \\label{tab:f_measure_%s}
\\end{table}
    """ % (node_id, node_id)
    
    return tex_string
