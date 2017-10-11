#Python imports
import os
import sys
import timeit

#Imports from the framework
import util

def extract(img_path, img_classes, param):
    """
    Function that performs the extraction of an image using the CEDD
    descriptor.
    
    This function transforms the image being extracted to the desired image
    format of the descriptor, performs the extraction of the image, and last,
    transforms the output feature vector of the descriptor into the standard
    output of the framework, a list of floats.
    """
    
    print "Descriptor: CEDD"
    
    #CONSTANTS
    #Number of executions of the extraction
    NUM_EXEC = 1
    
    #PATHS
    sys.path.append(os.path.dirname(__file__))
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
    img_name = list_classes + img_path.split(os.sep)[-1]
    
    #Convert the image to the desired format of the descriptor
    temp_img_path, converted = util.convert_desired_format(img_path, img_name,
                                                          "JPG")
    if converted:
        print "\tImage converted to JPG"
    
    #Path of the file with the feature vector of an image
    #CEDD does not need a file to write the feature vector
    
    #Extraction of the feature vector
    #-------------------------------------------------------------------------
    
    #Compile the .java files
    #   Change the current directory to the directory of the files
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    os.system("javac CEDD.java")
    print "\tJAVA files compiled"
    
    #Call subprocess jython
    jython_args = ['jython', 'jython_cedd.py', 'Extraction']
    jython_args.append(temp_img_path)
    jython_args.append(str(param))
    
    import subprocess
    cedd_process = subprocess.Popen(jython_args, stdout=subprocess.PIPE)
    print "\t", cedd_process
    cedd_extract = cedd_process.communicate()[0]
#    cedd_process.wait()
    fv = util.fv_string_to_list(cedd_extract)
    
    #   Return to the original directory
    os.chdir(orig_dir)
    #-------------------------------------------------------------------------
    
    #Remove the temporary image
    if converted:
        os.remove(temp_img_path)
    
    #Transforms the feature vector of the descriptor into the standard output
    #of the framework
    #Not necessary. The return of the getDoubleHistogram is already list of
    #floats
    
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
    import subprocess
    import xml.etree.cElementTree as ET
    from ast import literal_eval
    
    #Get parameters
    xml_name = os.path.join("Experiment " + experiment_id, "experiment.xml")
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                            "..", "experiments", xml_name))
    xml_file = ET.parse(xml_path)
    xml_desc = xml_file.findall("descriptor")
    for descriptor in xml_desc:
        if descriptor.attrib["name"] == "cedd":
            param = literal_eval(descriptor.get("parameters"))
            break
    
    #Change the current directory to the directory of the files
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    
    jython_args = ['jython', 'jython_cedd.py', 'Distance']
    jython_args.append(str(fv1))
    jython_args.append(str(fv2))
    jython_args.append(str(param))
    
    #Descriptor exclusive
    #-------------------------------------------------------------------------
    cedd_process = subprocess.Popen(jython_args, stdout=subprocess.PIPE)
    print "\t", cedd_process
    cedd_distance = cedd_process.communicate()[0]
    cedd_process.wait()
    distance = float(cedd_distance)
    
    #Return to the original directory
    os.chdir(orig_dir)
    #-------------------------------------------------------------------------
    
    return distance
