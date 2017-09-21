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
"""
Created on Fri Jun  7 16:58:24 2013

@author: waldir
"""
import config

import xml.etree.cElementTree as ET
import pydot

def create_graph_image(xml_path, img_path):
    """
    Create a visual representation of the experiment graph.
    
    This function reads the experiment's xml file and uses DOT/Graphviz to
    generate an image of the experiment's graph, which is then saved to a
    PDF file.
    
    Parameters
    ----------
        xml_path : string
            Path to the experiment's xml file.
        
        img_path : string
            Path to the pdf file that will store the resulting image.
        
    Returns
    -------
        None
    """
    
    xml = ET.parse(xml_path)

    graph = pydot.Dot(graph_type="digraph")
    graph.set_rankdir("LR")
     
    tags = ["collection", "descriptor", "normalizer", "classifier", \
            "evaluation_measure", "train_test_method", "fusion_method"]
    boxes = []
    for tag in tags:
        boxes.extend(xml.findall(tag))
    
    nodes = dict()
    for box in boxes:
        box_id = box.get("id")
        color = config.BLOCK_TYPE_STYLES[box.tag]["color2"]
        label = box.get("id") + "_" + box.get("name")
        nodes[box_id] = pydot.Node(label, style="filled", fillcolor=color, \
                                     fontsize="18", shape="box")
        graph.add_node(nodes[box_id])
        
    links = xml.find("links").findall("link")
    for link in links:
        source = link.get("id")
        for out_link in link.findall("out"):         
            target = out_link.text
            graph.add_edge(pydot.Edge(nodes[source], nodes[target]))
        
    graph.write_pdf(img_path)
    graph.write_png('{0}.png'.format(img_path[:-4]))
