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

import Tkinter as tk

import datetime
import glob
import os
from xml.etree import cElementTree as ElementTree
import appBlocks
import xml.dom.minidom
import config
import ast


def workflow_from_plugin_list(app, experiment_xml, workflow_xml):
    et = ElementTree.parse(experiment_xml)
    root = et.getroot()

    et2 = ElementTree.parse(workflow_xml)
    root2 = et2.getroot()

    experiment_name = root.attrib['id']
    experiment_author = root.attrib['author']
    experiment_number = root.attrib['number']
    experiment_date = root.attrib['date']
    experiment_hour = root.attrib['hour']
    experiment_iterations = root.attrib['iterations']
    
    #Get openset value and update the interface with the values of the
    # experiment
    experiment_openset = ast.literal_eval(root.attrib['openset'])
    
    app._experiment_name.set(experiment_name)
    app._author_name.set(experiment_author)
    app._num_iterations.set(experiment_iterations)
    if experiment_openset:
        app._openset_exp_check.select()

    blocks = []
    links = []

    for child in list(root):
        if child.tag in config.APPLICATION_BLOCKS:
            new_block = {}
            new_block['id'] = child.attrib['id']
            new_block['name'] = child.attrib['name']
            new_block['parameters'] = ast.literal_eval(child.attrib['parameters'])
            new_block['type'] = child.tag

            for child2 in list(root2):
                if child2.tag == child.tag:
                    if child2.attrib['id'] == child.attrib['id']:
                        new_block['x'] = child2.attrib['x']
                        new_block['y'] = child2.attrib['y']
                        break

            blocks.append(new_block)
        elif child.tag == "links":
            # Read links here
            for link in list(child):
                link_id = int(link.attrib['id'])
                # TODO: try catch above because of int() cast and in every
                # other cast below

                for link_direction in list(link):
                    if link_direction.tag == 'out':
                        curr_link = (link_id, int(link_direction.text))
                    elif link_direction.tag == 'in':
                        curr_link = (int(link_direction.text), link_id)
                    else:
                        curr_link = (None, None)

                    links.append(curr_link)
                        

    #Switch the context of the application according to the openset variable
    app.switch_openset_context()
    
    block_map = []

    for block in blocks:
        # TODO: try catch for int() cast below
        exp_block = appBlocks.ExperimentBlock(app=app, blocktype=block['type'],
                                              x=int(block['x']),
                                              y=int(block['y']))
        
        # TODO: look through every .xml in the folder given by block['type']
        # and find the plugin with name block['name']
        if block['type'] == config.COLLECTION:
            for elem in glob.glob(os.path.join(block['type'] + 's', "*.xml")):
                if read_collection_name(elem) == block['name']:
                    exp_block._context_box.switch_context(elem)
                    break
        else:
            exp_block._context_box.switch_context(os.path.join(block['type'] + 's', block['name'] + ".xml"))

        # Loading parameters
        parameters, parameter_types = exp_block._context_box.get_parameter_list()

        for key in block['parameters']:
            if key in parameters:
                if parameter_types[key] == 'range':
                    start, stop, step = block['parameters'][key].split(",")
                    parameters[key][0].set(start)
                    parameters[key][1].set(stop)
                    parameters[key][2].set(step)
                else:
                    parameters[key].set(block['parameters'][key])

        app.add_block(exp_block)
        block_map.append((block, exp_block))

    links = list(set(links))
    filled_links = []

    for curr_link in links:
        curr_filled_link = None
        first_block = None
        second_block = None

        for block in blocks:
            if block['id'] == str(curr_link[0]):
                first_block = block
                if second_block is not None:
                    break
            elif block['id'] == str(curr_link[1]):
                second_block = block
                if first_block is not None:
                    break

        for mapped_block in block_map:
            if first_block in mapped_block:
                first_block = mapped_block[1]
            if second_block in mapped_block:
                second_block = mapped_block[1]

        curr_filled_link = (first_block, second_block, True)
        filled_links.append(curr_filled_link)

    app.register_links(filled_links)


def xml_name(path):
    return path[path.rfind('/') + 1:-4]


def generate_experiment_xml(block_list, link_list, num_iterations,
                            experiment_name, author_name, openset):
    
    # Create folder with experiment id
    biggest_id = 0

    if not os.path.isdir('experiments'):
        os.mkdir('experiments')
    for fld in os.listdir('experiments'):
        if os.path.isdir(os.path.join('experiments', fld)):
            curr_id = int(fld[fld.find('_') + 1:])

            if curr_id > biggest_id:
                biggest_id = curr_id
    
    root = ElementTree.Element('experiment')
    root.set('date', datetime.date.today().strftime('%Y-%m-%d'))
    root.set('hour', datetime.datetime.now().strftime('%H:%M:%S'))
    root.set('number', str(biggest_id + 1))
    root.set('id', experiment_name)
    root.set('author', author_name)
    root.set('iterations', num_iterations)
    root.set('openset', str(bool(openset)))

    root_workflow = ElementTree.Element('workflow')
    root_workflow.set('date', datetime.date.today().strftime('%Y-%m-%d'))
    root_workflow.set('hour', datetime.datetime.now().strftime('%H:%M:%S'))
    root_workflow.set('number', str(biggest_id + 1))
    root_workflow.set('id', 'Giga-Framework-Test')

    nextID = 1
    block_ids = []

    for block in block_list:
        if block.blocktype in config.APPLICATION_BLOCKS:
            curDB = ElementTree.SubElement(root, block.blocktype)
            curDB.set('id', str(nextID))
            curDB.set('name', xml_name(block._context_box.plugin_name))
            curDB.set('parameters', str(block._context_box.get_parameters()))

            curDB_workflow = ElementTree.SubElement(root_workflow, block.blocktype)
            curDB_workflow.set('id', str(nextID))
            curDB_workflow.set('x', str(block.x))
            curDB_workflow.set('y', str(block.y))

            block_ids.append((block, nextID))

        nextID += 1

    links = ElementTree.SubElement(root, 'links')
    link_elements = []

    for i in xrange(1, nextID):
        link_elements.append(ElementTree.SubElement(links, 'link'))
        link_elements[i - 1].set('id', str(i))

    for link in link_list:
        b1 = link[0]
        b1_id = 0
        b2 = link[1]
        b2_id = 0

        for block in block_ids:
            if b1 in block:
                b1_id = block_ids[block_ids.index(block)][1]
            if b2 in block:
                b2_id = block_ids[block_ids.index(block)][1]

        cur_out = ElementTree.SubElement(link_elements[b1_id - 1], 'out')
        cur_out.text = str(b2_id)
        cur_in = ElementTree.SubElement(link_elements[b2_id - 1], 'in')
        cur_in.text = str(b1_id)

    curr_path_dir = os.path.join('experiments', 'Experiment_%s' % str(biggest_id + 1))
    os.mkdir(curr_path_dir)

    with open(os.path.join(curr_path_dir, 'experiment.xml'), 'w') as f:
        xmlf = xml.dom.minidom.parseString(
            ElementTree.tostring(root, encoding='utf-8'))
        f.write(xmlf.toprettyxml())

    # Write XML in Framework folder
    with open(os.path.join(curr_path_dir, 'workflow.xml'), 'w') as f:
        xmlf = xml.dom.minidom.parseString(
            ElementTree.tostring(root_workflow, encoding='utf-8'))
        f.write(xmlf.toprettyxml())

    return str(biggest_id + 1)


def get_collection_list():
    """Searches the collections directory and returns an alphabetically sorted
    list with all collection names."""
    collection_list = []

    for collection in glob.glob(os.path.join('collections', '*.xml')):
        collection_list.append((read_collection_name(collection), collection))

    collection_list = sorted(collection_list, key=lambda t: str.lower(t[0]))

    return collection_list


def get_plugin_name_list(plugin_type, openset_experiment):
    """Searches the directory of the specified type of plugin for plugins and
    returns a list sorted alphabetically of the names with both the name of
    the plugin and the name of the XML file."""
    plugin_list = []
    directory_name = None

    openset_only = False

    plugin_type = plugin_type.lower()

    if plugin_type == config.FEATURE:
        directory_name = "descriptors"
    elif plugin_type == config.CLASSIFIER:
        directory_name = "classifiers"
        openset_only = openset_experiment
    elif plugin_type == config.NORMALIZER:
        directory_name = "normalizers"
    elif plugin_type == config.COLLECTION:
        directory_name = "collections"
    elif plugin_type == config.EVALUATION:
        directory_name = "evaluation_measures"
        openset_only = openset_experiment
    elif plugin_type == config.TRAINTEST:
        directory_name = "train_test_methods"
    elif plugin_type == config.FUSION:
        directory_name = "fusion_methods"

    for elem in glob.glob(os.path.join(directory_name, "*.xml")):
        if openset_only:
            plugin_list.append((read_plugin_name(elem), elem))
        else:
            if is_openset_plugin(elem):
                continue
            plugin_list.append((read_plugin_name(elem), elem))

    plugin_list = sorted(plugin_list, key=lambda t: str.lower(t[0]))

    return plugin_list


def read_collection_name(file_name):
    """Returns the name of the collection XML."""
    et = ElementTree.parse(file_name)
    root = et.getroot()

    return root.attrib['id']


def is_openset_plugin(file_name):
    """Returns true if the openset tag is present in the plugin XML."""
    et = ElementTree.parse(file_name)
    root = et.getroot()

    if 'openset' in root.attrib:
        return root.attrib['openset'].lower() == 'true'

    return False


def read_plugin_name(file_name):
    """Returns the name of the plugin XML."""
    et = ElementTree.parse(file_name)
    root = et.getroot()

    return root.attrib['name']


def read_collection_description(file_name, contBlock):
    """Reads a collection XML and sends its data to a context block."""
    
    et = ElementTree.parse(file_name)
    root = et.getroot()

    collection_name = root.attrib['id']
    is_retrieval = False
    try:
        is_retrieval = root.attrib['retrieval']
    except:
        pass

    contBlock.add_title(collection_name)
    contBlock.add_label_with_linebreak(
        "Classes: %s" % root.attrib['number_classes'])
    contBlock.add_label_with_linebreak(
        "Instances: %s" % root.attrib['number_images'])
    contBlock.add_label_with_linebreak("List of classes")

    # If the experiment is openset, must add checkboxes for the user to select
    # positive classes
    if contBlock.app._is_openset_experiment:
        for child in list(root):
            if child.tag == 'class':
                contBlock.add_checkbox(name=child.attrib['id'])

        contBlock.add_integer(name="Number of classes", default='0')
    else:
        for child in list(root):
            if not is_retrieval:
                if child.tag == 'class':
                    contBlock.add_label_with_linebreak(child.attrib['id'])
            else:
                contBlock.add_label_with_linebreak("\tRetrieval experiment")
                break
    
    #Updates the scroll
    contBlock.configure(scrollregion=(0, 0, contBlock.CONTEXT_SIZE[0],
            contBlock._next_pos))


def read_plugin_description(file_name, contBlock):
    """Reads a plugin XML and sends its data to a context block."""
    et = ElementTree.parse(file_name)
    root = et.getroot()

    pluginName = root.attrib['name']

    contBlock.add_title(pluginName)

    for child in list(root):
        if child.tag == 'parameter':
            add_item(contBlock, child)
        elif child.tag == 'abletolink':
            contBlock.set_able_to_link(child.text)

def update_plugin_description(file_name, name_tag, contBlock):
    """Read a plugin XML and update its data to a context block."""
    et = ElementTree.parse(file_name)
    root = et.getroot()
    
    # Change the name_tag, replacing spaces with underline
    name_tag = name_tag.replace(' ', '_')

    child_list = []
    for child in list(root):
        if child.tag == 'parameter':
            child_list.append(child.get('name'))
            add_item(contBlock, child, name_tag)

    return child_list

def read_plugin_description_to_list(file_name):
    """Reads a plugin XML and returns three lists, respectively:
    * List of parameter widgets
    * List of control variables
    * List of control variables types"""
    et = ElementTree.parse(file_name)
    root = et.getroot()

    param_widgets = []
    param_cvs = []
    param_cvs_types = []

    pluginName = root.attrib['name']

    # add_title

    for child in list(root):
        if child.tag == 'parameter':
            # add_item
            pass

def add_item(contBlock, item, name_tag=None):
    itemType = item.get('type')
    name = item.get('name')
    default_value = item.get('default', None)
    fixed_value = item.get('fixed', False)
    reference_value = item.get('reference', None)

    if itemType == 'dropdown':
        options = []

        for option in item.findall('item'):
            options.append(option.text)

        contBlock.add_dropdown(name=name, items=options, default=default_value,
                               fixed=fixed_value, tags=name_tag,
                               reference=reference_value)
    elif itemType == 'integer':
        contBlock.add_integer(name=name, default=default_value,
                              fixed=fixed_value, tags=name_tag,
                              reference=reference_value)
    elif itemType == 'float':
        contBlock.add_float(name=name, default=default_value,
                            fixed=fixed_value, tags=name_tag)
    elif itemType == 'checkbox':
        contBlock.add_checkbox(name=name, default=default_value,
                               fixed=fixed_value, tags=name_tag)
    elif itemType == 'list':
        contBlock.add_list(name=name, default=default_value,
                           fixed=fixed_value, tags=name_tag)
    elif itemType == 'range':
        start = item.findall("./parameter[@name='Start']")[0]
        stop = item.findall("./parameter[@name='Stop']")[0]
        step = item.findall("./parameter[@name='Step']")[0]

        contBlock.add_range(name=name, start=start.get('type', 'integer'),
                            stop=stop.get('type', 'integer'),
                            step=step.get('type', 'integer'),
                            default_start=start.get('default', None),
                            default_stop=stop.get('default', None),
                            default_step=step.get('default', None),
                            fixed=fixed_value, tags=name_tag)
    elif itemType == 'filedialog':
        contBlock.add_filedialog(name=name, default_path=default_value,
                                 fixed=fixed_value, tags=name_tag)
    elif itemType == 'string':
        contBlock.add_string(name=name, default=default_value,
                             fixed=fixed_value, tags=name_tag,
                             reference=reference_value)
