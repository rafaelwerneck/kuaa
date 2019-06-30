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
    Function that performs the extraction of an image using the GCH
    descriptor.
    
    This function transforms the image being extracted to the desired image
    format of the descriptor, performs the extraction of the image, and last,
    transforms the output feature vector of the descriptor into the standard
    output of the framework, a list of floats.
    """
    
    print "Descriptor: GCH"
    
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
                plugin_name = 'gch_32l.so'
            else:
                plugin_name = 'gch_64l.so'
        else:
            plugin_name = 'gch_64l.so'
    
    #Extraction of the feature vector
    if not os.path.exists(fv_path):
        #Example:
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
    values = file_fv.read().split()[1:]
    for v in values:
        list_fv.append(float(v))
    
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
            plugin_name = 'gch_32l.so'
        else:
            plugin_name = 'gch_64l.so'
    else:
        plugin_name = 'gch_64l.so'

    plugin_path = os.path.join(descriptor_path, plugin_name)
    plugin = ctypes.CDLL(plugin_path)
    
    #Descriptor exclusive
    #-------------------------------------------------------------------------
    #Creating class requested by the Distance function
    class Histogram(ctypes.Structure):
        #Example: Pointer to a float vector and an integer
        _fields_ = [("v", ctypes.POINTER(ctypes.c_ubyte)),
                    ("n", ctypes.c_int)]
    
    #First Class
    Hist1 = Histogram()
    
    len_fv1 = len(fv1)
    fv1_int = map(int, fv1)
    c_v1 = (ctypes.c_ubyte * len_fv1)(*fv1_int)
    
    Hist1.v = ctypes.cast(c_v1, ctypes.POINTER(ctypes.c_ubyte))
    Hist1.n = ctypes.c_int(len_fv1)
    
    p_Hist1 = ctypes.pointer(Hist1)
    
    #Second Class
    Hist2 = Histogram()
    
    len_fv2 = len(fv2)
    fv2_int = map(float, fv2)
    c_v2 = (ctypes.c_double * len_fv2)(*fv2_int)
    
    Hist2.v = ctypes.cast(c_v2, ctypes.POINTER(ctypes.c_ubyte))
    Hist2.n = ctypes.c_int(len_fv2)
    
    p_Hist2 = ctypes.pointer(Hist2)
    
    #Parameters of the Distance function
    plugin.Distance.restype = ctypes.c_double
    #-------------------------------------------------------------------------
    
    #Execution
    distance = plugin.Distance(p_Hist1, p_Hist2)
    
    return distance
