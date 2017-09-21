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

import sys
import os
import xml.etree.cElementTree as ET
import ast

def verificate_item(links, type_name, actual_id):
    """
    Verification of every item of a experiment XML file.
    
    Every ID in the input link of this verification is searched in the 
    possibles links provided by the software plugin XML.
    
    Parameters
    ----------
        links : list, [ElementTree]
            List of ElementTree containing every link of the experiment XML.
            
        type_name : dict, {string: [string, string, string]}
            Dictionary containing all Elements from the XML different from
            links. Its format as {id: [tag, name, parameters]}
            
        actual_id : string
            String of the id being verified.
    
    Returns
    -------
        None
        
    """
    
    print "Verification of the id", actual_id
    
    #Every XMl file is inside a folder with name: tag + 's' 
    plugin_type = type_name[actual_id][0] + "s"
    path_plugin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", plugin_type))
    
    tag_name = type_name[actual_id][0]
    software_name = type_name[actual_id][1]
    item_parameters = ast.literal_eval(type_name[actual_id][2])
    
    #Open the specific XML file of the software that will be executed
    #posteriorly
    software_xml = ET.parse(os.path.join(path_plugin, software_name + ".xml"))
    list_abletolink = software_xml.findall('abletolink')
    list_parameters = software_xml.findall('parameter')
    list_possible = []
    parameters = []
    for able in list_abletolink:
        list_possible.append(able.text)
    for param in list_parameters:
        parameters.append(param.attrib['name'])
    
    #Visit every tag 'link' in the experiment XML to find the ID that is
    #been verified
    for link in links:
        if link.attrib['id'] == actual_id:
            list_inputs = link.findall('in')
            #Looks for every tag 'in' to verify if the link is permitted
            #by the framework
            for link_input in list_inputs:
                link_index = link_input.text.strip()
                if type_name[link_index][0] not in list_possible:
                    print "\nError in the tag", tag_name, "with name", \
                        software_name, "with id", actual_id, "\n"
                    sys.exit(1)
                for param in parameters:
                    if param not in item_parameters:
                        print "\nError. Parameter", param, \
                              "missing in the tag", tag_name, "with name", \
                              software_name, "with id", actual_id, "\n"
                        sys.exit(1)
            break
    
    print "End of the verification of the id", actual_id

def verification(xml_path):
    """
    Verification of the XML file with the experiment.
    
    For every step of the framework, is made a verification of the input links
    with the possibles links provided by the software XML.
    
    Parameters
    ----------
        xml_path : string
            Path to the experiment XML file.
    
    Returns
    -------
        None
        
    """
    
    print "\n#######################"
    print "\nVerification of the XML:", xml_path
    
    tree = ET.parse(xml_path)
    
    # Verify modules
    root = tree.getroot()
    type_name = {}
    for child in root:
        if child.tag != 'links':
            type_name[child.attrib['id']] = [child.tag, child.attrib['name'],
                                             child.attrib['parameters']]
    
    # Verify links
    links = tree.find('links')
    for actual_id in type_name:
        verificate_item(links, type_name, actual_id)
    
    print "End of verification"
    print "\n#######################\n\n"
