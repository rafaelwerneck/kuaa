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

# Python imports
import os
import sys
import argparse

def create_xml_collection(collection_path):
    folder = os.path.dirname(collection_path)
    name_file = collection_path.split(os.sep)[-1]
    xml_path = os.path.join(folder, name_file + ".xml")
    
    classes_dict = {}
    count_img = 0
    for class_dir in os.listdir(collection_path):
        class_path = os.path.join(collection_path, class_dir)
        if os.path.isdir(class_path):
            classes_dict[class_dir] = []
            for (dad_name, child_names, file_names) in os.walk(class_path):
                for obj in file_names:
                    obj_path = os.path.join(dad_name, obj)
                    if not os.path.isdir(obj_path) and obj != ".DS_Store" and obj != "ClipInfo.txt":
                        classes_dict[class_dir].append(obj_path)
                        count_img += 1
    
    #FILE
    print "Writing XML file"
    xml_file = open(xml_path, "w")
    xml_file.write("<?xml version=\"1.0\" ?>\n")
    xml_file.write("<collection id=\"%s\" number_classes=\"%d\" number_objects=\"%d\">\n" % (name_file, len(classes_dict.keys()), count_img))
    for class_item in classes_dict.keys():
        xml_file.write("\t<class id=\"%s\">\n" % (class_item))
        for obj in classes_dict[class_item]:
            xml_file.write("\t\t<object>%s</object>\n" % obj)
        xml_file.write("\t</class>\n")
    xml_file.write("</collection>")
    xml_file.close()
    print "Finished writing XML file\n"

if __name__ == "__main__":

    # Argument Parser
    parser = argparse.ArgumentParser(description="Collection XML Creator")
    parser.add_argument('collection_path', help="Path to the collection folder.")
    
    args = parser.parse_args()
    
    create_xml_collection(os.path.abspath(args.collection_path))
