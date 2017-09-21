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

#Future
from __future__ import division

#Python imports
import os
import sys
from datetime import datetime, timedelta
import xml.etree.cElementTree as ET

#Framework imports
import config

#Constants
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_CLASSES = 0
POS_FV = 1
INDEX_ZERO = 0

def main(collection_name, openset_experiment, parameters, node_id):
    """
    Read the paths and classes of a collection into memory.
    
    The paths and classes are read and returned as dictionaries.
    
    Parameters
    ----------
        collection_name : string
            Name of the collection.
        
        openset_experiment : boolean
            Boolean to determinate a open set experiment.
        
        parameters : dict, {string : string}
            Dictionary with the open set parameters.
            
        node_id : string
            ID of the collection node in the experiment.
        
    Returns
    -------       
        images : dict, {string : [list, list]}
            A dictionary where each (key, value) pair corresponds to the path to
            an item in the collection and a list structures indexed by the path,
            respectively.
            For example, images[path][0] is the list of classes associated with 
            that item.
        
        classes : dict, {string : [list of string]}
            A dictionary where each (key, value) pair corresponds to a class in
            the collection and a list of paths to the items associated with that
            class, respectively.
            
        extract_path : string
            Path to the folder where the feature extraction of the experiment will
            be stored.
            
        read_time : float
            Time taken to execute this function.
    """

    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Calculate the time to read the collection
    init_read = datetime.now()
    
    #Variables containing the values of the collection
    images = {}
    classes = {}
    
    #Paths
    collection_path = os.path.join(os.path.dirname(__file__), "..", "collections",
                                 collection_name + ".xml")
    
    extract_path = os.path.join(os.path.dirname(__file__), "..", "results",
                               collection_name) + os.sep

    #Create the directories, if they do not exist
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    #Read the XML file and save:
    ## Each class and the list of images
    ## Each image path and its classes
    collection_file = ET.parse(collection_path)
    list_class = collection_file.findall('class')
    total_classes = int(collection_file.getroot().attrib['number_classes'])
    num_classes = 0
    for item in list_class:
        name_class = item.attrib['id']
        list_images = []
        for child in item:
            image_path = child.text
            list_images.append(image_path)
            if image_path not in images:
                # TODO: better way to index multiple structures?
                images[image_path] = [[], []]
            images[image_path][POS_CLASSES].append(name_class)
        classes[name_class] = list_images
        num_classes += 1
        try:
            if num_classes % 10 == 0:
                socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                        (num_classes / total_classes)))
        except:
            pass
    
    print "\tNumber of Objects:", len(images.keys()), \
            "\n\tNumber of Classes:", len(classes.keys())
    
    #OpenSet
    #--------------------------------------------------------------------------
    if openset_experiment:
        classes_list = classes.keys()
        known_classes = []
            
        #List of Classes
        if parameters['Number of classes'] == 0:
            known_classes = []
            for item_param in parameters:
                if parameters[item_param]:
                    known_classes.append(item_param)
            
        else:
            from random import sample
            
            number_of_classes = int(parameters['Number of classes'])
            
            #Sample the classes list
            if number_of_classes >= len(classes_list):
                known_classes = classes_list
            else:
                known_classes = sample(classes_list, number_of_classes)
        
        #Change the label classes of the unknown classes to None
        new_images = {}
        new_classes = {}
        new_classes[None] = []
        
        print "Changing label of the unknown classes to None."
        print "Known Classes:", known_classes
        for img in images.iterkeys():
            list_classes = []
            img_fv = images[img][POS_FV]
            for item_class in images[img][POS_CLASSES]:
                if item_class not in known_classes:
                    try:
                        if not list_classes[INDEX_ZERO]:
                            list_classes.insert(INDEX_ZERO, None)
                    except:
                        list_classes.append(None)
                    list_classes.append(item_class)
                    if img not in new_classes[None]:
                        new_classes[None].append(img)
                else:
                    list_classes.append(item_class)
                    if item_class not in new_classes.keys():
                        new_classes[item_class] = []
                    if img not in new_classes[item_class]:
                        new_classes[item_class].append(img)
            new_images[img] = [list_classes, img_fv]
        
        images = new_images
        classes = new_classes
    #--------------------------------------------------------------------------
    
    try:
        socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id, 1.0))
    except:
        pass
    
    #Calculate the time to read the collection
    end_read = datetime.now()
    read_time = end_read - init_read
    read_time = read_time.total_seconds()
    print "Total time to read the collection: ", read_time, " seconds"
    
    return images, classes, extract_path, read_time
