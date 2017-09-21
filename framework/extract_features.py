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

#import from python library
import os
from datetime import datetime, timedelta
import multiprocessing
import sys
import socket
import shutil

#Import from other file
import util
import config

#Constants
BUFFER_LIMIT = 1E4
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_CLASSES = 0
INDEX_ZERO = 0

#Global variable
str_buffer = ""
extracted_path = ""
new_images = {}
number_images = 0
node_id = 0
total_images = 0

def main(images, classes_keys, extract_path, descriptor_name, param, id_node):
    """
    Call the descriptor plugin to extract the descriptors of the given images.
    
    Parameters
    ----------
        images : dict, {string : [list, list]}
            The keys of the dictionary are the paths to the images whose
            descriptors will be extracted.
        
        classes_keys : list
            List of the classes in the experiment.
            
        extract_path : string
            Path to the folder where the feature extraction of the experiment will
            be stored.
                    
        descriptor_name : string
            Name of the descriptor, which will be used to locate the plugin.
            
        param : dict, {string : string}
            Dictionary with the plugin-specific parameters. Usually extracted from
            the experiment's xml.
            
        id_node : string
            ID of the descriptor plugin node in the experiment.   
    
    Returns
    -------
        new_images : dict, {string : [list of string, [list of float]}
            Dictionary containing the list of classes and the feature vector 
            (list of float) of a given image, indexed by its path.
            
        extract_time : float
            Time taken to execute this function.
    
    """
    
    global str_buffer
    global extracted_path
    global new_images
    global node_id
    global total_images
    global number_images
    
    node_id = id_node
    number_images = 0
    
    #Send the start of the module
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Total of images being extracted
    total_images = len(images.keys())
    
    new_images = {}
    
    print "Extraction Module"
    
    #Calculate the extraction time
    init_extract = datetime.now()
    
    #Add the path to the descriptors to import the software of extraction
    descriptors_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "descriptors", descriptor_name))
    sys.path.append(descriptors_path)

    #The Python file for the descriptor can not have the same name of the
    #executable
    software = __import__("plugin_" + descriptor_name)
    print software
    
    #File where the collection of feature vectors will be saved
    extracted_path = os.path.join(os.path.dirname(extract_path), \
                                  descriptor_name + "." + str(param).replace(os.sep, '%') + ".fv")
    if not os.path.exists(os.path.dirname(extracted_path)):
        os.makedirs(os.path.dirname(extracted_path))
    
    #Read Features
    if descriptor_name == 'read_features':
        param["Extracted path"] = extracted_path
        _, _, _, _ = software.extract("", "", param)
        
    if os.path.exists(extracted_path):
        print "\tExtraction already executed. Moving on..."
        new_images = util.read_fv_file(extracted_path)
        
        complete_new_images = True
        
        images_to_remove = []
        for image in new_images.iterkeys():
            if image not in images.keys():
                images_to_remove.append(image)
            for item_class in new_images[image][POS_CLASSES]:
                if item_class not in classes_keys:
                    new_images[image][POS_CLASSES].insert(INDEX_ZERO, None)
                    break
        for image in images_to_remove:
            del new_images[image]
        
        for image in images.iterkeys():
            if image not in new_images:
                print image
                complete_new_images = False
                break
        
        if complete_new_images:
            end_extract = datetime.now()
        
            #Calculate the total extraction time
            extract_time = end_extract - init_extract
            extract_time = extract_time.total_seconds()
            print "Total extract time: ", extract_time, " seconds"
            
            #In case that the feature vectors are already extracted, send 100%
            try:
                socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id, 1.0))
            except:
                pass
            
            return new_images, extract_time
    
    if descriptor_name == "read_features":
        #Read file with features
        str_buffer = ""#software.extract(param)
    else:
        #number of cores to multiprocess the extraction
        num_cores = int(multiprocessing.cpu_count())
        print "Number of cores to be used: ", num_cores
        
        #run extractor
        pool = multiprocessing.Pool(num_cores)
        for img_path, value in images.iteritems():
            img_classes = value[POS_CLASSES]
            if img_path not in new_images:
                pool.apply_async(software.extract,
                                 args = (img_path, img_classes, param, ),
                                 callback = save_buffer)
        pool.close()
        pool.join()
    
    extracted_file = open(extracted_path, "ab")
    extracted_file.write(str_buffer)
    extracted_file.close()
    
    #Clean buffer
    str_buffer = ""
    
    #End of the extraction
    print "Success of the extraction"
    try:
        socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id, 1.0))
    except:
        pass
    
    end_extract = datetime.now()
    
    #Calculate the total extraction time
    extract_time = end_extract - init_extract
    extract_time = extract_time.total_seconds()
    print "Total extract time: ", extract_time, " seconds"
    
    return new_images, extract_time

def save_buffer(result):
    """
    Buffer a fv and save to disk when the buffer is full.    
    
    Callback of the Pool.apply_async, receive the result of the operation made 
    by the software.extract and add it in the buffer. In case the buffer exceed
    the BUFFER_LIMIT, put it in the normalized_file and clean the buffer.
    
    Parameters
    ----------
        result : list, [img_path_name, len(img_classes), img_classes, fv]
            Used to save a feature vector to disk in the framework's format.
    
    Returns
    -------
        None
    """
    
    global str_buffer
    global extracted_path
    global new_images
    global number_images
    global node_id
    global total_images
    
    print "Get result from process"
    img_path_result, img_classes_len, img_classes_result, img_fv_result = \
            result
            
    #Seve new images
    new_images[img_path_result] = [img_classes_result, [img_fv_result]]
    
    #To write into the file, ignore the class None
    img_classes_result = img_classes_result[1:] if not \
            img_classes_result[INDEX_ZERO] else img_classes_result
    
    print "\tSave result in the buffer"
    str_buffer = str_buffer + img_path_result + " " + \
            str(len(img_classes_result)) + " " + str(img_classes_result) + \
            " " + str(img_fv_result)
    #In case that the buffer is greater than BUFFER_LIMIT, save the buffer
    #in the extracted_path
    if sys.getsizeof(str_buffer) >= BUFFER_LIMIT:
        print "Saving the buffer into the file..."
        extracted_file = open(extracted_path, "ab")
        extracted_file.write(str_buffer)
        extracted_file.close()
        str_buffer = ""
    str_buffer = str_buffer + "\n"
    
    number_images += 1
    try:
        socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                (number_images / total_images)))
    except:
        pass
