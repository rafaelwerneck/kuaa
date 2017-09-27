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

#Python Imports
import __builtin__
import os
import sys
import xml.etree.cElementTree as ET
import ast
import copy
import socket
import subprocess
from datetime import datetime
from multiprocessing import Manager

#Framework imports
import util
import xml_verification
import read_collection
import train_test
import extract_features
import extract_bag
import normalize_features
import classify
import fusion
import evaluation
import config
import experiment_graph

#Constants
HOST = config.COMMUNICATION_HOST
PORT = config.COMMUNICATION_PORT
END_EXPERIMENT = config.MESSAGE_EXPERIMENT_FINISH
END_ITERATION = config.MESSAGE_ITERATION_FINISH
POS_CLASSES = 0
INDEX_ZERO = 0

# Global variables
execution_time = 0
fusion_dict = {}
tex_path = ""
folder_name = "" # folder where intermediate/temporary files are stored
tex_dict = {}
openset_experiment = False

def execute(node, previous, experiment_folder):
    """
    Execute a task defined by the given node in the experiment graph.
    
    Parameters
    ----------
    
    node : Element
        The node to be executed.
        
    previous : dict (or list of dict)
        Dictionary of the experiment's running-time variables after the
        end of the parent node's execution.
        May be a list of dictionaries in the special case of a fusion node,
        which has more than one parent.
    
    experiment_folder : string
        String with the path to the experiment folder, where the files of the
        experiment will be saved.
        
    Returns
    -------
    
    exp_param : dict
        The updated dictionary of the experiment's running-time variables after
        the node's execution.
    
    """

    global execution_time
    global tex_path
    global tex_dict
    global openset_experiment

    exp_param = previous
    parameters = ast.literal_eval(node.get("parameters"))
    node_id = node.attrib['id']
    
    #Get node name
    node_name = node.get('name')

    if node.tag == "collection":
        print "Collection", exp_param.keys()
        
        images, classes, extract_path, read_time = \
                read_collection.main(node_name, openset_experiment, parameters,
                node_id)
        execution_time += read_time

        exp_param['images'] = images
        exp_param['classes'] = classes
        exp_param['extract_path'] = extract_path

    elif node.tag == "train_test_method":
        print "train_test_method", exp_param.keys()
    
        images = exp_param['images']
        classes = exp_param['classes']

        images, classes, train_test_list, train_test_time = \
                train_test.main(images, classes, experiment_folder, node_name,
                parameters, openset_experiment, node_id)
        execution_time += train_test_time

        exp_param['images'] = images
        exp_param['classes'] = classes
        exp_param['train_test_list'] = train_test_list
        
        exp_param['train_test_method'] = node_name
        exp_param['train_test_parameters'] = parameters

    elif node.tag == "descriptor":
        print "descriptor", exp_param.keys()
    
        images = exp_param['images']
        extract_path = exp_param['extract_path']
        classes_keys = exp_param['classes'].keys()
        
        if node_name == "bag":
            train_test_list = exp_param['train_test_list']
            
            images, extract_time = extract_bag.main(images, train_test_list,
                    extract_path, experiment_folder, parameters, node_id)
            
        elif node_name == "bovg":
            train_test_list = exp_param['train_test_list']
            
            images, extract_time = extract_bovg.main(images, train_test_list,
                    extract_path, experiment_folder, parameters, node_id)
            
        else:
            images, extract_time = extract_features.main(images, classes_keys,
                    extract_path, node_name, parameters, node_id)
            
        execution_time += extract_time

        exp_param['images'] = images
        exp_param['descriptor'] = node_name

    elif node.tag == "normalizer":
        try:
            manager = Manager()
            images = manager.dict(exp_param['images'])
            train_test_list = exp_param['train_test_list']
        except:
            print "\n\tMissing Input. Exiting."
            sys.exit(1)
            
        norm_fv_paths, normalize_time = normalize_features.main(images,
                train_test_list, experiment_folder, node_name, parameters,
                node_id)
        execution_time += normalize_time

        del exp_param['images']
        exp_param['fv_paths'] = norm_fv_paths

    elif node.tag == "classifier":
        try:
            classes = exp_param['classes']
            train_test_list = exp_param['train_test_list']
            descriptor = exp_param['descriptor']
            try:
                fv_paths = exp_param['fv_paths']
                del exp_param['fv_paths']
            except:
                images = exp_param['images']
                fv_paths = util.save_file_extract(images, train_test_list,
                        experiment_folder)
        except:
            print "\n\tMissing Input. Exiting."
            sys.exit(1)
        
        images, classes_list, classify_time = classify.main(fv_paths,
                classes.keys(), train_test_list, experiment_folder, node_name,
                parameters, descriptor, node_id)
        execution_time += classify_time

        exp_param['images'] = images
        exp_param['classes_list'] = classes_list

    elif node.tag == "fusion_method":
        len_exp_param = len(exp_param)
        #list with the images dictionaries, classes dictionaries, and train and
        # test set list
        list_images = []
        list_classes = []
        list_train_test = []
        extract_path = exp_param[INDEX_ZERO]['extract_path']
        
        for index in range(len_exp_param):
            try:
                list_images.append(exp_param[index]['images'])
            except:
                images = {}
                for fv_path in exp_param[index]['fv_paths']:
                    print "fv_path:", fv_path
                    images_new = util.read_fv_file(fv_path)
                    images = util.merge_dict(images, images_new)
                list_images.append(images)
        
            list_classes.append(exp_param[index]['classes'])
            #In case that it performs the fusion of collections, there is no
            # train_test_list
            try:
                list_train_test.append(exp_param[index]['train_test_list'])
            except:
                list_train_test.append(None)
        #classes_list is present only after the classification module
        try:
            classes_list = exp_param[INDEX_ZERO]['classes_list']
        except:
            classes_list = None
        try:
            train_test_method = exp_param[INDEX_ZERO]['train_test_method']
            train_test_parameters = exp_param[INDEX_ZERO]['train_test_parameters']
        except:
            train_test_method = None
            train_test_parameters = None
        
        images, classes, train_test_list, fusion_time = \
                fusion.main(list_images, list_classes, list_train_test,
                        classes_list, experiment_folder, node_name, parameters,
                        node_id)
        execution_time += fusion_time
        
        exp_param = {}
        exp_param['images'] = images
        exp_param['classes'] = classes
        if train_test_list is not None:
            exp_param['train_test_list'] = train_test_list
        if classes_list is not None:
            exp_param['classes_list'] = classes_list
        if train_test_method is not None:
            exp_param['train_test_method'] = train_test_method
            exp_param['train_test_parameters'] = train_test_parameters
        exp_param['descriptor'] = None
        exp_param['extract_path'] = extract_path

    elif node.tag == "evaluation_measure":
        try:
            images = exp_param['images']
            train_test_list = exp_param['train_test_list']
            classes_list = exp_param['classes_list']
        except:
            print "\n\tMissing Input. Exiting."
            sys.exit(1)

        evaluation_time, evaluation_path = evaluation.main(images,
                train_test_list, classes_list, experiment_folder, node_name,
                parameters, node_id)
        execution_time += evaluation_time
        
        #Dictionaries to create the tex file
        train_test_method = exp_param['train_test_method']
        train_test_parameters = str(exp_param['train_test_parameters'])
        
        if train_test_method not in tex_dict:
            tex_dict[train_test_method] = {}
        train_test_dict = tex_dict[train_test_method]
        
        if train_test_parameters not in train_test_dict:
            train_test_dict[train_test_parameters] = {}
        output_dict = train_test_dict[train_test_parameters]
        
        if node_name not in output_dict:
            output_dict[node_name] = []
        list_output = [evaluation_path, classes_list[0], node_id]
        if list_output not in output_dict[node_name]:
            output_dict[node_name].append(list_output)
            
        train_test_dict[train_test_parameters] = output_dict        
        tex_dict[train_test_method] = train_test_dict
    
    elif node.tag == "preprocessing":
        images = exp_param['images']
        classes = exp_param['classes']
        
        images, classes, preprocessing_time = preprocessing.main(images,
                classes, experiment_folder, node_name, parameters, node_id)
        execution_time += preprocessing_time
        
        exp_param['images'] = images
        exp_param['classes'] = classes

    else:
        print "Error. Unknown Tag."
        sys.exit(1)

    return exp_param

def process_node(xml, node, previous, experiment_folder):
    """
    Process a single node in an experiment graph, and recurse on the children.
    
    This function identifies child nodes, executes the given node if its
    dependencies are finished (such as in fusion nodes), and recurses on the
    child nodes.
    
    Parameters
    ----------
    xml : ElementTree
        The experiment tree as a parsed xml object.
        
    node : Element
        The specific node to be processed.
        
    previous : dict
        Dictionary of the experiment's running-time variables after the
        end of the parent node's execution. Iniatially empty.
    
    experiment_folder : string
        String with the path to the experiment folder, where the files of the
        experiment will be saved.
    
    Returns
    -------
    None
    
    """

    # This dictionary's (key, value) pairs are node IDs, and the experiment
    # variables after finishing the referred node's execution, 
    # respectively. This is made global in order to enable it to be accessible 
    # by different executions of this function.
    # TODO: Check for possible efficiency issues. 
    # TODO: Using classes/OO could make this much more readable.
    global fusion_dict

    print "Processing node:", node

    # Find the id of the processing node
    links = xml.find("links").findall("link")
    for l in links:
        if l.get("id") == node.get("id"):
            link = l
            break
    # Find the id of all <in> of this link
    list_in_links = []
    for in_link in link.findall("in"):
        # gets the linked-node's id
        list_in_links.append(in_link.text.split()[0])
    # Find the id of all <out> of this link
    list_out_links = []
    for out_link in link.findall("out"):
        # gets the linked-node's id
        list_out_links.append(out_link.text.split()[0])

    print "TAG:", node.tag
    # In case that the tag is a fusion_method, wait until all input were done
    if node.tag == "fusion_method":
        list_previous = []
        for node_in in list_in_links:
            if node_in not in fusion_dict:
                print "Find Fusion Method. Missing input, going back in the", \
                      "execution."
                return
            list_previous.append(fusion_dict[node_in])
        previous = list_previous

    # Save the experiment variables up to the executed node.
    fusion_dict[node.get("id")] = execute(node, previous, experiment_folder)

    if not list_out_links:
        return
    else:
        for element in xml.getroot().getchildren():
            if element.get("id") in list_out_links:
                # Call the list function to make a copy, and not reference
                # to the same memory space
                process_node(xml, element,
                             copy.deepcopy(fusion_dict[node.get("id")]),
                             experiment_folder)

def run_experiment(xml_path):
    """
    Run an experiment defined in an xml file.
    
    This function is the starting point for running an experiment. It parses
    the xml file, starts a socket communication with the GUI, starts processing
    the root nodes in the experiment graph, and writes the results.
    
    Parameters
    ----------
    xml_path : string
        Path to the xml file defining the experiment.
        
    Returns
    -------
    None    
    
    """

    # Append directory framework to sys.path
    sys.path.insert(0, os.path.dirname(__file__))

    global execution_time
    global tex_path
    global tex_dict
    global fusion_dict
    global openset_experiment

    xml_verification.verification(xml_path)

    # Socket for sending the progress of the module
    try:
        __builtin__.socket_framework = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_framework.connect((HOST, PORT))
    except:
        pass

    xml = ET.parse(xml_path)
    __builtin__.experiment_id = xml.getroot().get("number")
    openset_experiment = ast.literal_eval(xml.getroot().get("openset"))
    node_dic = {}
    for child in xml.getroot():
        if child.tag != 'links':
            node_dic[child.attrib['id']] = child
    print node_dic

    #Graph image
    graph_img_path = os.path.join(os.path.dirname(xml_path), "graph.pdf")
    experiment_graph.create_graph_image(xml_path, graph_img_path)

    # .TEX FILE
    #--------------------------------------------------------------------------
    # Create the .tex file to show the results of the execution
    tex_path = os.path.abspath(os.path.join(os.path.dirname(xml_path),
                                            "experiment.tex"))
    experiment_folder = os.path.abspath(os.path.dirname(xml_path)) + os.sep

    tex_file = open(tex_path, "w")
    
    #Get header of the experiment
    header_name = xml.getroot().get("id")
    header_author = xml.getroot().get("author")

    # Write the head of a article tex file
    tex_file.write("""
\\documentclass[a4paper,10pt]{article}

\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
\\usepackage{morefloats}
\\usepackage{lscape}

\\catcode`_=12
\\begingroup\\lccode`~=`_\\lowercase{\\endgroup\\let~\\sb}
\\mathcode`_="8000

\\title{%s}
\\author{%s}

\\begin{document}

\\maketitle

\\section{Experiment Workflow}
\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{"%s"}
\\end{figure}
    """ % (header_name, header_author, os.path.abspath(graph_img_path[:-4])))

    tex_file.close()
    #--------------------------------------------------------------------------

    print "-----     -----     -----"
    print "Experiment", experiment_id
    
    for __builtin__.iteration in range(int(xml.getroot().attrib['iterations'])):
        fusion_dict = {}
        links = xml.find("links").findall("link")
        for link in links:
            link_in = False
            for child in link:
                if child.tag == 'in':
                    link_in = True
            if not link_in:
                print "Executing workflow number:", iteration + 1
                process_node(xml, node_dic[link.attrib['id']], {},
                        experiment_folder)

        print "\nTotal execution time: ", execution_time, " seconds"
        
        try:
            socket_framework.sendall("%s %s///" % (END_ITERATION, ""))
        except:
            pass
    
    print "\nAdding the evaluation measures to the tex file"
    output_tex = ""
    print tex_dict
    for train_test_method in tex_dict.keys():
        for train_test_parameters in tex_dict[train_test_method]:
            param = ast.literal_eval(train_test_parameters)
            output_tex += train_test.write_tex(train_test_method, param)
            output_dict = tex_dict[train_test_method][train_test_parameters]
            for output_method in output_dict:
                output_tex += evaluation.write_tex(output_method,
                                               output_dict[output_method])

    # .TEX FILE
    #--------------------------------------------------------------------------
    tex_file = open(tex_path, "a")

    tex_file.write("""
%s
\\end{document}
    """ % (output_tex))

    tex_file.close()

    # Generate the .pdf from the .tex file
    orig_dir = os.getcwd()
    os.chdir(os.path.dirname(tex_path))
    print "\npdflatex", tex_path
    subprocess.call(["pdflatex", tex_path])
    subprocess.call(["pdflatex", tex_path])
    os.chdir(orig_dir)
    #--------------------------------------------------------------------------

    # Send to socket the end of the experiment
    try:
        socket_framework.sendall("%s %s///" % (END_EXPERIMENT,
                                               tex_path[:-3] + "pdf"))
        socket_framework.close()
    except:
        pass
