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
    
#CONSTANTS
POS_CLASSES = 0
POS_PREDICT = 1
INDEX_ZERO = 0

def evaluation(images, test_set, classes_list, pos_train_test, parameters):
    """
    Performs the calculation of a Confusion Matrix
    
    Calculate the Confusion Matrix using the entries of a given file and save
    the result in the evaluation file.
    """
    
    print "\tOutput: Confusion Matrix"
    
    #Get parameters
    
    #Output methods
    #-------------------------------------------------------------------------
    list_class = []
    result_class = []
    
    for img_test in test_set:
        list_class.append(str(images[img_test][POS_CLASSES][INDEX_ZERO]))
        classes_proba = images[img_test][POS_PREDICT][pos_train_test]
        img_predict_index = INDEX_ZERO
        for i in range(len(classes_list)):
            if classes_proba[i] > classes_proba[img_predict_index]:
                img_predict_index = i
        result_class.append(str(classes_list[img_predict_index]))
    
    #Calculate the confusion matrix
    cm = metrics.confusion_matrix(numpy.array(list_class),
                                  numpy.array(result_class),
                                  classes_list)
    
    #-------------------------------------------------------------------------
    print "Success in the calculation of the Confusion Matrix"
    
    #Delete variables that will not be used
    del list_class
    del result_class
    
    return cm

def string_file(cm):
    """
    Creates a one line confusion matrix to be write in the file.
    """
    
    cm_write = []
    for row in cm:
        cm_write.append(list(row))
    
    return str(cm_write)

def tex_name():
    """
    Return a latex section with the name of the evaluation.
    """
    
    print "\tConfusion Matrix"
    return "\\newpage\n\\subsection{Confusion Matrix}"

def write_tex(evaluation_path, classes, node_id):
    """
    Create a Confusion Matrix from the confusion matrices in the evaluation_path.
    """
    
    from ast import literal_eval
    import pylab
    import numpy
    
    print "\t\tTeX: Confusion Matrix"
    
    len_classes = len(classes)
    
    max_len_classes = max([len(str(x)) for x in classes])
    
    cm_list = []
    
    evaluation_file = open(evaluation_path, "rb")
    for line in evaluation_file.readlines():
        cm_list.append(numpy.array(literal_eval(line)))
    evaluation_file.close()
    
    sum_cm = sum(cm_list)
    
    #Normalize Confusion Matrix
    norm_cm = []
    for row in sum_cm:
        sum_row = 0
        temp_array = []
        sum_row = sum(row, 0)
        for column in row:
            temp_array.append(column / sum_row)
        norm_cm.append(temp_array)
    norm_cm = numpy.array(norm_cm)
    
    #Write in the evaluation file the sum_cm and the norm_cm
    evaluation_file = open(evaluation_path, "ab")
    evaluation_file.write("\nSum of the Confusion Matrices\n")
    evaluation_file.write(str(sum_cm))
    evaluation_file.write("\n\nNormalized Confusion Matrix\n")
    evaluation_file.write(str(norm_cm))
    evaluation_file.close()
    
    #Using PyLab, create a visual representation of the confusion
    #   matrix and save it
    pylab.clf()
    figure = pylab.figure()
    axes = figure.add_subplot(111)
    axes.set_aspect(1)
    res = axes.imshow(norm_cm, cmap=pylab.cm.jet, interpolation='nearest',
                      vmin=0.0, vmax=1.0)
    pylab.title("Confusion Matrix")
    pylab.xticks(range(len_classes), classes[:len_classes])
    pylab.yticks(range(len_classes), classes[:len_classes])
    for tick in pylab.gca().xaxis.iter_ticks():
        tick[0].label1.set_rotation(90)
    figure.tight_layout()
    figure.colorbar(res)
    file_fig = os.path.join("experiments", "Experiment_" + experiment_id,
                            str(node_id) + "_confusion_matrix")
    pylab.savefig(file_fig + ".pdf")
    
    tex_string = """
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\columnwidth]{"%s"}
    \\caption{Average Confusion Matrix of Node %s.}
    \\label{fig:cm_%s}
\\end{figure}
    """ % (os.path.abspath(file_fig), node_id, node_id)
    
    tex_string += """
\\begin{landscape}
\\begin{table}[htbp]
    \\centering
    \\begin{tabular}{%s}
        %s \\\\
        \\hline
    """ % ('c|' + 'c'*(len_classes), '&' + ' & '.join('\\rotatebox{90}{' + str(x) + '}' for x in classes))
    
    for index, value in enumerate(norm_cm):
        tex_string += """
        %s &%s\\\\""" % (classes[index],
                         ' & '.join("{:.2f}".format(x * 100) \
                                for x in norm_cm[index]))
    
    tex_string += """
    \\end{tabular}
    \\caption{Average Confusion Matrix of Node %s.}
    \\label{tab:tab_%s}
\\end{table}
\\end{landscape}
    """ % (node_id, node_id)
    
    return tex_string
