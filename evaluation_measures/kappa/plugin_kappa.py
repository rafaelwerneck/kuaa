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
from __future__ import division
from sklearn import metrics
import numpy

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of the Cohen's Kappa.
    
    Cohen's kappa coefficient is a statistical measure of inter-annotator
    agreement for qualitative items.
    
    To calculate thsi coefficient, firstly is calculated the confusion matrix
    with all classes. After, kappa is calculated by the formula
        (Pr(a) - Pr(e))/(1 - Pr(e)),
    where Pr(a) is the relative observed agreement among raters, and Pr(e) is
    the hypothetical probability of chance agreement.
    """
    
    #CONSTANTS
    POS_CLASSES = 0
    POS_PREDICT = 1
    
    print "\tOutput: Cohen's Kappa"
    
    #Output methods
    #-------------------------------------------------------------------------
    list_class = []
    result_class = []
    
    number_classes = len(classes_list)
    number_images = len(test_set)
    
    for img_test in test_set:
        list_class.append(images[img_test][POS_CLASSES][0])
        classes_proba = images[img_test][POS_PREDICT][pos_train_test]
        img_predict_index = 0
        for i in range(number_classes):
            if classes_proba[i] > classes_proba[img_predict_index]:
                img_predict_index = i
        result_class.append(classes_list[img_predict_index])
    
    #Calculate the confusion matrix
    cm = metrics.confusion_matrix(numpy.array(list_class),
                                  numpy.array(result_class),
                                  classes_list)
    
    #Relative Aggrement
    relative_sum = 0.0
    for i in range(number_classes):
        relative_sum += cm[i][i]
    relative_aggrement = relative_sum / number_images
    
    #Hypothetical Aggrement
    hypothetical_list = []
    hypothetical_aggrement = 0.0
    for i in range(number_classes):
        row = 0
        for j in range(number_classes):
            row += cm[i][j]
        row = row / number_images
        
        column = 0
        for j in range(number_classes):
            column += cm[j][i]
        column = column / number_images
        
        hypothetical_list.append(row * column)
    
    for i in range(number_classes):
        hypothetical_aggrement += hypothetical_list[i]
    
    #Kappa
    kappa = (relative_aggrement - hypothetical_aggrement) / (1.0 - hypothetical_aggrement)

    #-------------------------------------------------------------------------
    print "Success in the calculation of the Cohen's Kappa"
    
    #Delete variables that will not be used
    del list_class
    del result_class
    del number_classes
    del number_images
    del relative_sum
    del relative_aggrement
    del hypothetical_list
    del hypothetical_aggrement
    
    return kappa

def string_file(kappa):
    """
    Returns a one line Cohen's Kappa to be write into the file.
    """
    
    return str(kappa)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tCohen's Kappa"
    return "\\subsection{Cohen's Kappa}"

def write_tex(evaluation_path, classes, node_id):
    """
    Calculates the average Cohen's Kappa from the values in the evaluation_path.
    """
    
    from numpy import array
    from scipy import stats
    from math import sqrt
    
    print "\t\tTeX: Cohen's Kappa"
    
    kappa_list = []
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        kappa_list.append(float(line))
    evaluation_file.close()
    
    avg_kappa = array(kappa_list).mean()
    std_kappa = array(kappa_list).std()
    interval = stats.t.interval(0.95, len(kappa_list) - 1, loc = avg_kappa,
            scale = std_kappa / sqrt(len(kappa_list)))
    conf = avg_kappa - interval[0]
    
    evaluation_file = open(evaluation_path, "ab")
    evaluation_file.write("\nAverage Cohen's Kappa\n")
    evaluation_file.write(str(avg_kappa))
    evaluation_file.write("\nStandard Deviation\n")
    evaluation_file.write(str(std_kappa))
    evaluation_file.write("\nConfidence Interval (95%)\n")
    evaluation_file.write(str(conf))
    evaluation_file.close()
    
    tex_string = """
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{ccc}
        Mean & Deviation & Confidence Interval (95\\%%)\\\\
        \\hline
        %.2f & %.2f & %.2f
    \\end{tabular}
    \\caption{Average, Standard Deviation and Confidence Interval of the Cohen's Kappa of Node %s}
    \\label{tab:kappa_%s}
\\end{table}
    """ % (avg_kappa * 100, std_kappa * 100, conf * 100, node_id, node_id)
    
    return tex_string
