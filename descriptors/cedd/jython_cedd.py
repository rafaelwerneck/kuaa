import sys
from ast import literal_eval
import array

from java.io import File
from javax.imageio.ImageIO import read
import CEDD

def Extraction(img_path, parameters):
    #Get parameters
    parameters = literal_eval(parameters)
    th0 = parameters['T0']
    th1 = parameters['T1']
    th2 = parameters['T2']
    th3 = parameters['T3']
    compact = parameters['CompactDescriptor']
    
    #Read image
    j_image = File(img_path)
    j_buffered = read(j_image)
    
    # Create CEDD object and extract the feature vector
    cedd = CEDD(th0, th1, th2, th3, compact)
    cedd.extract(j_buffered)
    
    #To get the result of the jython execution, print it
    fv = cedd.getDoubleHistogram().tolist()
    print fv

def Distance(fv1, fv2, parameters):
    #Get parameters
    parameters = literal_eval(parameters)
    th0 = parameters['T0']
    th1 = parameters['T1']
    th2 = parameters['T2']
    th3 = parameters['T3']
    compact = parameters['CompactDescriptor']
    
	# Transform the string feature vector in a list
    list_fv1 = fv_string_to_list(fv1)
    list_fv2 = fv_string_to_list(fv2)
    cedd_fv1 = array.array('d', list_fv1)
    cedd_fv2 = array.array('d', list_fv2)
    
    #Create CEDD objects
    cedd1 = CEDD(th0, th1, th2, th3, compact, cedd_fv1)
    cedd2 = CEDD(th0, th1, th2, th3, compact, cedd_fv2)

	# Calculate distance between two CEDD feature vectors
    print cedd1.getDistance(cedd2)

def fv_string_to_list(string):
    """
    Function originate from the 'util.py', that cannot be imported by
    the process jython.
    """
    
    fv = []
    
    fv = string.split()
    fv = ','.join(fv)
    fv = fv[2:-2]
    fv = fv.split(',')
    fv = map(float, fv)
    
    return fv

if __name__ == "__main__":
    if sys.argv[1] == 'Extraction':
        #img_path, parameters
        Extraction(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'Distance':
        #fv1, fv2, parameters
        Distance(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(1)
