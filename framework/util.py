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
import os
import sys
import pickle
from ast import literal_eval
import re

try:
    import Image #PIL to convert image
except:
    from PIL import Image

#Constants
POS_CLASSES = 0
POS_FV = 1
INDEX_ZERO = 0

def convert_desired_format(img_path, img_name, desired_format):
    """
    Convert the image to the desired format of the descriptor passed
    as parameter
    
    Parameters
    ----------
        img_path : string
            Path to the image being converted.
            
        img_name : string
            String with the name to save the converted image.
            
        desired_format : string
            String with the extension of the desired converted image.
    
    Returns
    -------
        img_final_path : string
            Path to the converted image, or the original image path.
            
        converted : boolean
            Boolean if the image was converted or not.
    
    """
    #if the image is not in the desired format, converts it
    if not img_name.upper().endswith(desired_format.upper()):

        try:
            im = Image.open(img_path)
        except:
            print "\tERROR opening the image", img_path
            sys.exit(1)

        if (desired_format.upper() != "PGM"):
            im = im.convert("RGB") #color image
        else:
            im = im.convert("L")   #gray image

        #saves the converted version to a temp directory
        #(does not change the original image!)
        img_final_path = os.path.join(os.path.dirname(__file__), "..", "temp",
                img_name.split(".")[0] + "." + desired_format.lower())
        im.save(img_final_path) 
        converted = True
    else:
        converted = False
        img_final_path = img_path

    return img_final_path, converted

def read_fv_file(file_path):
    """
    Function to read the file and returns a dictionary with the images path and
    its feature vectors.
    
    Parameters
    ----------
        file_path : string
            Path to the file containing the feature vectors of the images of
            the database.
    
    Returns
    -------
        new_images : dict, {string : [list, list]}
            Dictionary containing, for each image path, the classes of the
            image and the feature vectors.
    """
    
    BUFFER_LIMIT = int(1E9)
    
    new_images = {}
    
    fv_file = open(file_path, "rb")
    lines = fv_file.readlines(BUFFER_LIMIT)
    while lines != []:
        for line in lines:
            line = line.split()
            
            end_of_img_path = 0
            for index in range(len(line)):
                try:
                    value = int(line[index])
                    end_of_img_path = index - 1
                except:
                    pass
            
            img_path = " ".join(line[0:end_of_img_path + 1])
            img_num_classes = int(line[end_of_img_path + 1])
            
            img_classes = line[end_of_img_path + 2 : end_of_img_path + 2 + img_num_classes]
            img_classes = ''.join(img_classes)
            img_classes = literal_eval(img_classes)
            
            fv = line[end_of_img_path + 2 + img_num_classes : ]
            fv = ''.join(fv)
            fv = fv[1:-1]
            fv = fv.split(',')
            fv = map(float, fv)
            
            new_images[img_path] = [img_classes, [fv]]
        lines = fv_file.readlines(BUFFER_LIMIT)
    fv_file.close()
    
    return new_images

def save_file_extract(images, train_test_list, experiment_folder):
    """
    Create the files of the extraction in case that the experiment does not
    normalize the feature vectors before the classification.
    
    Parameters
    ----------
        images : dict, {string : [list, list]}
            Dictionary containing the classes and feature vectors for each
            image of the experiment.
            
        train_test_list : list of list
            List of the datasets splits to be used in the experiment, with 
            each entry containing the training and testing sets.
            
        experiment_folder : string
            Path to the folder to save the files of the extraction.
    
    Returns
    -------
        fv_paths : list
            List of the files with the feature vector.
        
    """
    
    print "Util: Save File Extract"
    
    fv_paths = []
    
    extract_path = experiment_folder + "iteration:" + str(iteration) + \
            "-extraction-train_test_"
    for index, train_test in enumerate(train_test_list):
        train, test = train_test
        extract_path_index = extract_path + str(index) + ".txt"
        extract_file = open(extract_path_index, "wb")
        for image_path in train + test:
            image_classes = images[image_path][POS_CLASSES]
            try:
                image_fv = images[image_path][POS_FV][index]
            except:
                image_fv = images[image_path][POS_FV][INDEX_ZERO]
            extract_file.write(image_path + " " + str(len(image_classes)) + \
                    " " + str(image_classes) + " " + str(image_fv) + "\n")
        extract_file.close()
        fv_paths.append(extract_path_index)
    
    return fv_paths

def merge_dict(dict1, dict2):
    """
    Merges dictionaries of images.
    """
    
    for key in dict2.iterkeys():
        if key not in dict1:
            dict1[key] = dict2[key]
        else:
            dict1[key][POS_FV].append(dict2[key][POS_FV][INDEX_ZERO])
    
    return dict1

def fv_string_to_list(string):
    """
    Read a string and return a list of floats.
    """
    
    fv = []
    
    fv = string.split()
    fv = ''.join(fv)
    fv = fv[1:-1]
    fv = fv.split(',')
    fv = map(float, fv)
    
    return fv
    
def load_collection(file_name):

    #loads the collection file
    collection_file = open(file_name,"r")
    collection = pickle.load(collection_file)
    collection_file.close()
    return collection

def get_collection_part(collection_size, current_part, num_cores):

    #breaking the collection into parts and taking only the desired part
    print "extracting part ", current_part, " of ", num_cores
    part_size = int(collection_size / num_cores)
    ini_index = (current_part - 1) * part_size
    if (current_part < num_cores):  #not the last part yet
        end_index = current_part * part_size
    else:
        end_index = collection_size

    return ini_index, end_index
    
def list_to_string(list):
    #Create a list to store the feature vector from the file
    string = ""
    
    for item in list:
        string = string + str(item) + " "
    string = string[:-1]
        
    return string

def get_software(descriptor):
    """
    Import the descriptor module from the descriptors folder and call the
    function 'get_software'
    """
    
    path_descriptors = os.path.abspath(os.path.join("..", "descriptors"))
    sys.path.append(path_descriptors)
    
    command = __import__("d_" + descriptor)
    return (command.get_software(), command.get_descriptor_path(),
            command.get_img_format())

def standard_output(descriptor, string):
    """
    Transforms the string output of the descriptor in the standard output
    of the framework
    """
    
    path_descriptors = os.path.abspath(os.path.join("..", "descriptors"))
    sys.path.append(path_descriptors)
    
    command = __import__("d_" + descriptor)
    return command.fv_transform(string)

def line_fv(line):
    """
    Split the line in image name, image class, and feature vector
    """
    
    line = line.split()
    
    img_name = line[0]
    img_class = line[1]
    
    fv = line[2:] #Get the feature vector from the line
    fv = ''.join(fv) #As the list is compost by strings contain the itens 
                     #from the feature vector, concatenate the itens
    fv = fv[1:-1] #Eliminate the brackets
    fv = fv.split(',') #Create a list of the itens of the feature vector,
                       #eliminating the comma
    fv = map(float, fv) #As the itens of the feature vector are still
                        #strings, map them to float

    return img_name, img_class, fv

def read_array(string):
    """
    Read string that contais a numpy array.
    """
    
    search_array = "array\("
    search_end = "\)"
    find_array = re.finditer(search_array, string)
    find_end = re.finditer(search_end, string)
