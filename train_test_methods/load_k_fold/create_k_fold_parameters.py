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

import sys
import os

def main(number_of_folds):
    """
    Create a XML file containing a sequence of input files for each fold.
    """
    
    number_of_folds = int(number_of_folds)
    
    # Text from the XML file
    xml_text = """
<software name="Create Folds">"""
    
    for i in range(number_of_folds):
        xml_text += """    
    <parameter type="filedialog" name="Training Fold {0}" />
    <parameter type="filedialog" name="Testing Fold {0}" />""".format(i)
    
    xml_text += """
</software>"""
    
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
            "folds_{0}.xml".format(number_of_folds)))
    
    xml_file = open(xml_path, 'w')
    xml_file.write(xml_text)
    xml_file.close()
    
    return xml_path
