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

#Python imports
from datetime import datetime, timedelta
import os
import multiprocessing
from ast import literal_eval
import sys
import glob

#Framework imports
import config
import util

#Global Variables
START = config.MESSAGE_MODULE_START
PROGRESS = config.MESSAGE_MODULE_PROGRESS
POS_CLASSES = 0
POS_FV = 1
POS_TRAIN = 0
POS_TEST = 1
ZERO_INDEX = 0

new_images = {}
list_low_level = []
number_images = 0
total_images = 0
node_id = 0
list_coding = []
list_pooling = []
list_images_remove = []

def main(images, train_test_list, extract_path, experiment_folder, parameters,
        id_node):
    """
    
    Parameters
    ----------
        images : dict, {string : [list, list]}
            The keys of the dictionary are the paths to the images whose
            descriptors will be extracted.
            
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            For example, train_test_list[i][POS_TRAIN] is a list of paths of items
            in the training set for the i'th split, while
            train_test_list[i][POS_TEST] is the testing set for the same split.
            
        extract_path : string
            Path to the folder where the feature extraction of the experiment will
            be stored.
        
        experiment_folder : string
            String with the path to the experiment folder, where the files of the
            experiment will be saved.
            
        parameters : dict, {string : string}
            Dictionary with the plugin-specific parameters. Usually extracted from
            the experiment's xml.
            
        id_node : string
            ID of the descriptor plugin node in the experiment.
    
    Returns
    -------
        new_images : dict, {string : [list of string, [list of float]}
            Dictionary containing the list of classes and the feature vector 
            (list of float) of a given image, indexed by its path.
            
        bag_time : float
            Time taken to execute this function.
    
    """
    
    global new_images
    global list_low_level
    global number_images
    global total_images
    global node_id
    global list_coding
    global list_pooling
    global list_images_remove
    
    #Global
    new_images = {}
    number_images = 0
    num_images = len(images.keys())
    total_images = num_images + len(train_test_list) * (num_images + 1)
    node_id = id_node
    list_low_level = []
    list_coding = []
    list_pooling = []
    
    #Communication with the interface
    try:
        socket_framework.sendall("%s %s///" % (START, node_id))
    except:
        pass
    
    #Get Parameters
    sampling_method = parameters["Sampling"]
    num_words = parameters["Number of Words"]
    quantization = parameters["Quantization"].lower()
    coding = parameters["Coding"]
    pooling = parameters["Pooling"]
    #Plugins name
    coding_name = plugin_coding(coding)
    pooling_name = plugin_pooling(pooling)
    
    print "\tSampling", sampling_method
    print "\tWords", num_words
    print "\tQuantization", quantization
    print "\tCoding", coding
    print "\tPooling", pooling
    
    new_images = {}
    
    print "Extraction Bag of Visual Words"
    
    #Calculate the extraction time
    bag_init = datetime.now()
    
    #Get number of cores to multiprocess
    num_cores = int(multiprocessing.cpu_count())
    
    #File where the collection of feature vectors will be saved
    extracted_path = os.path.join(os.path.dirname(experiment_folder),
            'iteration:' + str(iteration) + "-" + str(parameters) + "_")
    if len(extracted_path) > 225:
        extracted_path = extracted_path.replace(" ", "")
    bag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                             "descriptors", "bag"))
    
    #Low-level feature extraction
    #--------------------------------------------------------------------------
    #Path to the keys
    low_level_keys = os.path.abspath(os.path.join(
            os.path.dirname(extract_path), "bag." + str(parameters) + ".keys"))
            
    #List of images that the features are already extracted
    already_extracted = []
            
    #In case that the file with the low level extraction exists, the file with
    # all keys are splitted in a file for each image, and the low level
    # extraction is skipped
    if os.path.exists(low_level_keys):
        keys_file = open(low_level_keys, "rb")
        #Read the header of the file
        header1 = keys_file.readline()
        # Ignore the next two lines
        keys_file.readline()
        keys_file.readline()
        header4 = keys_file.readline()
        header5 = keys_file.readline()
        
        #Paths
        dir_path = os.path.dirname(__file__)
        temp_path = os.path.join(dir_path, "..", "temp")
        temp_path = os.path.abspath(temp_path)
        
        img_file_path = ""
        list_names = []
        list_write = []
        
        for line in keys_file:
            line = line.split()
            img_name = line[1]
            
            #Look-up if the img_name is in the train or test
            for train, test in train_test_list:
                if img_name in train + test:
                    already_extracted.append(img_name)
                    break
            #In case that the img_name is not found, ignore line in keys_file
            else:
                continue
                
            str_write = " ".join(line[1:])
            if img_name not in list_names:
                if img_file_path != "":
                    #Changed the img_name, save the img file
                    len_write = len(list_write)
                    file_img = open(img_file_path, "wb")
                    file_img.write(header1)
                    file_img.write(str(1).ljust(12) + "\n")
                    file_img.write(str(len_write).ljust(12) + "\n")
                    file_img.write(header4)
                    file_img.write(header5)
                    for index, item in enumerate(list_write):
                        file_img.write(str(index + 1) + " " + item + "\n")
                    file_img.close()
                    list_write = []
                    append_low_level((img_file_path, img_name))
                #Get classes of the img_name
                img_name_classes = images[img_name][POS_CLASSES]
                list_classes = ""
                for name_class in img_name_classes:
                    list_classes += str(name_class) + "_"
                
                img_name_split = img_name.split(os.sep)
                file_name = list_classes + img_name_split[-2] + "_" + \
                        img_name_split[-1] + ".fv"
                img_file_path = os.path.join(temp_path, file_name)
                list_names.append(img_name)
            list_write.append(str_write)
        #EOF, write the last file
        len_write = len(list_write)
        file_img = open(img_file_path, "wb")
        file_img.write(header1)
        file_img.write(str(1).ljust(12) + "\n")
        file_img.write(str(len_write).ljust(12) + "\n")
        file_img.write(header4)
        file_img.write(header5)
        for index, item in enumerate(list_write):
            file_img.write(str(index + 1) + " " + item + "\n")
        file_img.close()
        list_write = []
        append_low_level((img_file_path, img_name))
        keys_file.close()

    #Get unique values of image_path extracted
    already_extracted = list(set(already_extracted))
            
    print "Performing the low-level extraction"
    
    #Add the path to the plugin of the sampling method
    sampling_path = os.path.abspath(os.path.join(bag_path,
            sampling_method))
    sys.path.append(sampling_path)
    
    software = __import__("plugin_" + sampling_method)
    print software
    
    pool = multiprocessing.Pool(num_cores)
    print "\tNumber of Images: {0}\n\tAlready extracted: {1}\n".format(len(images.keys()), len(already_extracted))
    for img_path, value in images.iteritems():
        if img_path not in already_extracted:
            img_classes = value[POS_CLASSES]
            pool.apply_async(software.extract, args = (img_path, img_classes, parameters, ), callback = append_low_level)
            
    pool.close()
    pool.join()
    
    # Remove images that has no points to extract.
    for image_path_remove in list_images_remove:
        print "\tDelete {0}".format(image_path_remove)
        del images[image_path_remove]
        for train, test in train_test_list:
            try:
                train.remove(image_path_remove)
                break
            except:
                pass
            try:
                test.remove(image_path_remove)
                break
            except:
                pass
    
    #Compile all keys into one file
    num_images = num_images - len(list_images_remove)
    print "{0}\t{1}\n".format(num_images, len(list_images_remove))
    
    # Empty list of images to remove.
    list_images_remove = []
    
    if len(list_low_level) != num_images:
        print "Error! Number of low level files different from number of files! {0} != {1}\n".format(len(list_low_level), num_images)
        sys.exit(1)
    software.compile_keys(list_low_level, low_level_keys,
            parameters)
    
    #Remove the temporary files with the parameters and the result of the
    #   extraction
    for low_level_file in list_low_level:
        file_list = glob.glob(low_level_file + ".params*")
        for f in file_list:
            os.remove(f)
    #--------------------------------------------------------------------------
    print "len", len(train_test_list)
    for index, train_test in enumerate(train_test_list):
        list_coding = []
        list_pooling = []
        train = train_test[0]
        test = train_test[1]
    #Feature space quantization
    #--------------------------------------------------------------------------
        #File with the created dictionary
        dictionary_name = "iteration:" + str(iteration) + "_bag." + \
                str(parameters) + ".dictionary_" + str(index)
        if len(dictionary_name) > 255:
            dictionary_name = "".join(dictionary_name.split())
        dictionary_path = os.path.abspath(experiment_folder + dictionary_name)
        
        sys.path.append(os.path.join(bag_path, quantization))
        software = __import__("plugin_" + quantization)
        print software
        
        software.quantization(low_level_keys, train, num_words,
                              dictionary_path)
        
        number_images += 1
        try:
            socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                    (number_images / total_images)))
        except:
            pass
    #--------------------------------------------------------------------------
    
    #Coding and Pooling
    #--------------------------------------------------------------------------
        #Path to the coding and pooling
        coding_path = os.path.abspath(experiment_folder + "-iteration:" + \
                str(iteration) + "_bag." + str(parameters) + ".coding_" + \
                str(index))
        pooling_path = os.path.abspath(experiment_folder + "-iteration:" + \
                str(iteration) + "_bag." + str(parameters) + ".pooling_" + \
                str(index))
        
        sys.path.append(os.path.join(bag_path, coding_name))
        software_coding = __import__("plugin_" + coding_name)
        sys.path.append(os.path.join(bag_path, pooling_name))
        software_pooling = __import__("plugin_" + pooling_name)
        
        print "Coding:", software_coding, "\nPooling:", software_pooling
        
        pool = multiprocessing.Pool(num_cores)
        for low_level_file in list_low_level:
            print "low_level_file", low_level_file
            pool.apply_async(software_coding.coding,
                             args = (dictionary_path, low_level_file, parameters, num_words, pooling_name, ),
                             callback = coding_to_pooling)
        pool.close()
        pool.join()
        
        print "End Coding and Pooling."
        
        read_pooling_file(list_pooling, images, extracted_path,
                train_test_list)
    #--------------------------------------------------------------------------
        
        #Remove files from Coding and Pooling
        for low_level_file in list_low_level:
            file_list = glob.glob(low_level_file + ".*")
            for f in file_list:
                os.remove(f)
    
    #Remove all temporary files
    for low_level_file in list_low_level:
        file_list = glob.glob(low_level_file + "*")
        for f in file_list:
            os.remove(f)
    
    #Calculate the extraction time
    bag_end = datetime.now()
    bag_time = bag_end - bag_init
    bag_time = bag_time.total_seconds()
    print "Total extract time: ", bag_time, " seconds"
    
    return new_images, bag_time

def append_low_level(result):
    """
    Append the path to the result of the low-level extraction to the list of
    low-levels file.
    
    Callback of the Pool.apply_async, receive the result of the operation made 
    by the software.extract, append this result to the list of low-levels file
    and increase the percentage of work done.
    
    Parameters
    ----------
        result : string, fv_path
            Result append in the list of low-levels file, to be used in the compile
            keys function.
    
    Returns
    -------
        None
    """
    
    global list_low_level
    global number_images
    global total_images
    global node_id
    global list_images_remove
    
    print "\tGet result from low-level extraction"
    
    # In case that the descriptor didn't find any point to describe, create a list of this images to remove them.
    low_level, image_path = result
    
    if low_level == None:
        list_images_remove.append(image_path)
        print "\tImages Remove: {0}\n".format(len(list_images_remove))
    else:
        list_low_level.append(low_level)
        print "\tLow Level: {0}\n".format(len(list_low_level))
    
    number_images += 1
    try:
        socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                (number_images / total_images)))
    except:
        pass

def plugin_coding(coding):
    """
    Get the plugin name to performs the import.
    
    As that is needed the path to the coding function, the module needs to
    translate the path to the plugin to the name of the plugin file.
    
    Parameters
    ----------
        coding : string
            Path to the coding plugin in the framework.
    
    Returns
    -------
        coding_plugin : string
            String with the name of the plugin file.
    """
    
    if coding == "hard_assignment":
        return "hard_assignment"
    elif coding == "soft_assignment":
        return "soft_assignment"
    else:
        print "Error! Wrong coding parameter."
        sys.exit(1)

def plugin_pooling(pooling):
    """
    Get the plugin name of the pooling.
    
    Used to translate the user-friendly string name of the pooling plugin to
    the string interpreted by the framework.
    
    Parameters
    ----------
        pooling : string
            User-friendly string name of the pooling plugin.
    
    Returns
    -------
        pooling_plugin : string
            String with the name of the plugin file.
    """
    
    if pooling == "Max Pooling":
        return "max_pooling"
    elif pooling == "Average Pooling":
        return "avg_pooling"
    else:
        print "Error! Wrong pooling parameter."
        sys.exit(1)

def read_pooling_file(list_pooling, images, extracted_path, train_test_list):
    """
    Read the Pooling file, get the image and feature vector, and update the
    new_images dictionary.
    After that, create one file for each train and test set with the feature
    vectors of the images.
    
    Parameters
    ----------
        list_pooling : list
            List with the path with the pooling result of each image in the
            experiment.
        
        images : dict, {string : [list, list]}
            The keys of the dictionary are the paths to the images whose
            descriptors will be extracted.
            
        extracted_path : string
            Path to the file where the bag of visual words feature extraction
            will be stored.
        
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            For example, train_test_list[i][POS_TRAIN] is a list of paths of
            items in the training set for the i'th split, while
            train_test_list[i][POS_TEST] is the testing set for the same split.
    
    Returns
    -------
        None
    """
    
    global new_images
    
    print "Reading Pooling file"
    
    for index, pooling_path in enumerate(list_pooling):
    
        pooling_file = open(pooling_path, "rb")
        
        #Read header
        pooling_file.readline()
        pooling_file.readline()
        pooling_file.readline()
        pooling_file.readline()
        pooling_file.readline()
        
        #Read body
        for line in pooling_file.readlines():
            list_fields = line.strip().split()
            #Fields: id img_path x y a b c fv
            img_path = list_fields[1]
            img_classes = images[img_path][POS_CLASSES]
            fv = map(float, list_fields[7:])
            
            if img_path not in new_images:
                new_images[img_path] = [img_classes, [fv]]
            else:
                new_images[img_path][POS_FV].append(fv)
        
        pooling_file.close()
    
    for index, train_test in enumerate(train_test_list):
        train, test = train_test
        
        #Open feature file to write the feature vectors
        try:
            extracted_file = open(extracted_path + str(index), "wb")
        except:
            pass
        
        for img_path in train + test:
            img_classes = new_images[img_path][POS_CLASSES]
            try:
                img_fv = new_images[img_path][POS_FV][index]
            except:
                img_fv = new_images[img_path][POS_FV][ZERO_INDEX]
                
            extracted_file.write(img_path + " " + str(len(img_classes)) + \
                    " " + str(img_classes) + " " + str(img_fv) + "\n")
        
        extracted_file.close()
    
    print "\tSize of images:", len(new_images.keys())
    
    print "Ended reading Pooling file"

def coding_to_pooling(result):
    """
    Gather the coding result to performs the pooling step.
    
    For each image in the experiment, after the coding step, call the pooling
    plugin and append its result (the path to the pooling file) in the list of
    pooling paths.
    
    Parameters
    ----------
        result : list, [coding_path, num_words, pooling_name]
            Result of the coding step, made by the path of the coding file, the
            number of words and the name of the pooling plugin.
    
    Results
    -------
        None
    """
    
    global list_pooling
    global number_images
    global total_images
    global node_id
    
    print "\tGet result from coding and begin pooling."
    coding_path = result[0]
    num_words = result[1]
    pooling_name = result[2]
    
    pooling_software = __import__("plugin_" + pooling_name)
    
    pooling_path = pooling_software.pooling(num_words, coding_path)
    
    list_pooling.append(pooling_path)
    
    os.remove(coding_path)
    
    number_images += 1
    try:
        socket_framework.sendall("%s %s %f///" % (PROGRESS, node_id,
                (number_images / total_images)))
    except:
        pass
