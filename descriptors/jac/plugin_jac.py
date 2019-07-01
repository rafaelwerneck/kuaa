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
import timeit
import platform

#Imports from the framework
import util

def extract(img_path, img_classes, param):
    """
    Function that performs the extraction of an image using the JAC
    descriptor.
    
    This function transforms the image being extracted to the desired image
    format of the descriptor, performs the extraction of the image, and last,
    transforms the output feature vector of the descriptor into the standard
    output of the framework, a list of floats.
    """
    
    print "Descriptor: JAC"
    
    #CONSTANTS
    #Number of executions of the extraction
    NUM_EXEC = 1
    
    #PATHS
    #Path for the folder containing the descriptor executable
    ##Not necessary in case that the extraction is present in the plugin code
    descriptor_path = os.path.dirname(__file__)
    #Path for the temporary folder, where the file with the feature vector
    #extracted from the image will be saved
    ##Consider the path with the origin on the directory of the plugin
    ##(os.path.dirname(__file__)) to avoid problems with the directory where
    ##the framework is being executed
    path_temp = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
    if not os.path.isdir(path_temp):
        os.makedirs(path_temp)
    
    print img_path, "being extract in the process ", os.getpid()
    
    #Temporary name of the image in the desired image format
    list_classes = ""
    for name_class in img_classes:
        list_classes += str(name_class) + "_"
    list_names = img_path.split(os.sep)
    img_name = list_classes + list_names[-2] + "_" + list_names[-1]
    
    #Convert the image to the desired format of the descriptor
    temp_img_path, converted = util.convert_desired_format(img_path, img_name,
                                                      "PPM")
    if converted:
        print "\tImage converted to PPM"
    
    #Path of the file with the feature vector of an image
    fv_path = os.path.join(path_temp, img_name + ".fv")
    
    # Extraction of the feature vector
    if not os.path.exists(fv_path):
        system_platform = [platform.system(), platform.architecture()[0]]
        if system_platform[0] == 'Linux':
            if system_platform[1] == '32bit':
                plugin_name = 'jac_32l.so'
            else:
                plugin_name = 'jac_64l.so'
        else:
            plugin_name = 'jac_64l.so'
    
    #Extraction of the feature vector
    if not os.path.exists(fv_path):
        setup = """
ctypes = __import__('ctypes')
plugin = "%s"
lib = ctypes.CDLL("%s%s" + plugin)
img_path = "%s"
fv_path = "%s"
        """%(plugin_name, descriptor_path, os.sep, temp_img_path, fv_path)
        
        cmd = """
lib.Extraction(img_path, fv_path)
        """
        
        t = timeit.Timer(stmt=cmd, setup=setup)
        try:
            t.timeit(number=NUM_EXEC)
            print "\tFeature vector extracted"
        except:
            t.print_exc()
    
    #Remove the temporary image
    if converted:
        os.remove(temp_img_path)
    
    #Transforms the feature vector of the descriptor into the standard output
    #of the framework
    fv = fv_transform(fv_path)
    
    return img_path, len(img_classes), img_classes, fv

def fv_transform(fv_path):
    """
    Receive the path with the feature vector in the descriptor output and
    return the feature vector in the framework standard output, a list of
    floats.
    """
    
    list_fv = []
    
    #Open the file created by the descriptor, save the feature vector in the 
    #standard output, remove the file and return the new feature vector
    try:
        file_fv = open(fv_path, "rb")
    except IOError:
        print "ERROR"
        sys.exit(1)
    
    #Performs the necessary operations to transform the feature vector into
    #the standard output
    # nColors, nTexturedness, nGradient, nRank, maxDistance);    
    params = file_fv.readline().split()
    n = int(params[0]) * int(params[1]) * int(params[2]) * int(params[3]) * \
        int(params[4])
    
    values = file_fv.readline().split()
    k = 0
    values_size = len(values) - 2 # original vector has -1 -1 as last values

    for i in range(n):
        if k < values_size and int(values[k]) == i:
            list_fv.append(float(values[k+1]))
            k += 2
        else:
            list_fv.append(0.0)
                      
    file_fv.close()
    os.remove(fv_path)
    
    print "\tFeature vector transformed in the standard output"

    return list_fv
    
def distance(fv1, fv2):
    """
    Performs the calculation of distance between the two feature vectors,
    according to the Distance function of the executable.
    
    Inputs:
        - fv1 (list of floats): First feature vector
        - fv2 (list of floats): Second feature vector
    
    Output:
        - distance (double): Distance between the two feature vectors
    """
    
    #Imports
    import ctypes
    
    descriptor_path = os.path.dirname(__file__)

    system_platform = [platform.system(), platform.architecture()[0]]
    if system_platform[0] == 'Linux':
        if system_platform[1] == '32bit':
            plugin_name = 'jac_32l.so'
        else:
            plugin_name = 'jac_64l.so'
    else:
        plugin_name = 'jac_64l.so'

    plugin_path = os.path.join(descriptor_path, plugin_name)
    plugin = ctypes.CDLL(plugin_path)
    
    #Descriptor exclusive
    #-------------------------------------------------------------------------
    #Creating class requested by the Distance function
    class JointAutoCorrelogram(ctypes.Structure):
        #Example: Pointer to a float vector and an integer
        _fields_ = [("h", ctypes.POINTER(ctypes.c_float)),
                    ("nColors", ctypes.c_int),
                    ("nGradient", ctypes.c_int),
                    ("nTexturedness", ctypes.c_int),
                    ("nRank", ctypes.c_int),
                    ("maxDistance", ctypes.c_int)]
    
    #All Classes
    import xml.etree.cElementTree as ET
    import ast
    xml_name = os.path.join("Experiment " + experiment_id, "experiment.xml")
    xml_path = os.path.abspath(os.path.join(descriptor_path, "..", "..", "experiments", xml_name))
    xml_file = ET.parse(xml_path)
    xml_desc = xml_file.findall("descriptor")
    for descriptor in xml_desc:
        if descriptor.attrib["name"] == "jac":
            parameters = ast.literal_eval(descriptor.get("parameters"))
            num_colors = int(parameters["Colors"])
            num_gradients = int(parameters["Gradients"])
            num_texturedness = int(parameters["Texturedness"])
            num_ranks = int(parameters["Ranks"])
            num_distances = int(parameters["Distances"])
            break
    
    print num_colors, num_gradients, num_texturedness, num_ranks, num_distances
    
    #First Class
    JAC1 = JointAutoCorrelogram()
    
    len_fv1 = len(fv1)
    fv1_float = map(float, fv1)
    c_h1 = (ctypes.c_float * len_fv1)(*fv1_float)
    
    JAC1.h = ctypes.cast(c_h1, ctypes.POINTER(ctypes.c_float))
    JAC1.nColors = ctypes.c_int(num_colors)
    JAC1.nGradient = ctypes.c_int(num_gradients)
    JAC1.nTexturedness = ctypes.c_int(num_texturedness)
    JAC1.nRank = ctypes.c_int(num_ranks)
    JAC1.maxDistance = ctypes.c_int(num_distances)
    
    p_JAC1 = ctypes.pointer(JAC1)
    
    #Second Class
    JAC2 = JointAutoCorrelogram()
    
    len_fv2 = len(fv2)
    fv2_float = map(float, fv2)
    c_h2 = (ctypes.c_float * len_fv2)(*fv2_float)
    
    JAC2.h = ctypes.cast(c_h2, ctypes.POINTER(ctypes.c_float))
    JAC2.nColors = ctypes.c_int(num_colors)
    JAC2.nGradient = ctypes.c_int(num_gradients)
    JAC2.nTexturedness = ctypes.c_int(num_texturedness)
    JAC2.nRank = ctypes.c_int(num_ranks)
    JAC2.maxDistance = ctypes.c_int(num_distances)
    
    p_JAC2 = ctypes.pointer(JAC2)
    
    #Parameters of the Distance function
    plugin.Distance.restype = ctypes.c_double
    #-------------------------------------------------------------------------
    
    #Execution
    distance = plugin.Distance(p_JAC1, p_JAC2)
    
    return distance
