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

#Python import
import sys
import os
from datetime import datetime
import argparse

#Framework import
from framework import run_experiment

def initFramework(xml_path):

    # Get datetime
    date_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    
    experiment_dir_path = os.path.abspath(os.path.dirname(xml_path))

    log_file = open(os.path.join(experiment_dir_path, "output_{0}.log".format(date_time)), "w")
    sys.stdout = log_file #Change the standard output to the file log_file
    error_file = open(os.path.join(experiment_dir_path, "error_{0}.log".format(date_time)), "a")
    sys.stderr = error_file
    
    del date_time
    
    run_experiment.run_experiment(xml_path)
    
    sys.stdout = sys.__stdout__ #Reset the standard output
    sys.stderr = sys.__stderr__
    log_file.close()
    error_file.close()

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Kuaa Framework")
    parser.add_argument('xml_path', help="Path to the XML experiment file.")
    
    args = parser.parse_args()
    
    initFramework(args.xml_path)
