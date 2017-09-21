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

from collections import namedtuple
import Tkinter as tk
import tkFileDialog
import tkFont

import sys

import Image
import ImageTk

import config
from appUtils import *
from appBlocks import *
from interface.xmlUtils import workflow_from_plugin_list
from interface.xmlUtils import get_plugin_name_list
from interface.xmlUtils import get_collection_list
import socket_communication
from framework import recommendation

import copy
import math
import subprocess
import os
import threading

RECOMMENDATION_SHOW = 5
ROW_RADIO = 0
COLUMN_RADIO_BUTTON = 0
ROW_BUTTON = RECOMMENDATION_SHOW - 1
COLUMN_CHECKBUTTON = 1
COLUMN_RECOMMENDATION = 2

class FrameworkThread(threading.Thread):
    def __init__(self, experiment_id):
        threading.Thread.__init__(self, name="Framework Thread")
        self._experiment_id = experiment_id

    def run(self):
        subprocess.call("python " + "initFramework.py" + " " +
                        os.path.join("experiments", "Experiment_" +
                                     self._experiment_id, "experiment.xml"),
                        shell=True)


class InterfaceApp(tk.Tk):
    """Entry point to the interface application."""

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Creating main window
        self.canvas = tk.Canvas(width=self.winfo_screenwidth(),
                                height=self.winfo_screenheight(),
                                bg=config.STAGE_BACKGROUND, borderwidth=0,
                                highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Helper variables for every function in the interface
        self._button_press_position = {'x': 0, 'y': 0}
        self._afterjob = None
        self._wheel_visible = False
        self._link_mode = False
        self._current_link_block = None
        self._link_deleted = False
        self._currentMousePositionPair = None
        self._link_cleared = True
        self._top_bar_visible = False
        self._experiment_running = False
        self._is_openset_experiment = False

        self._wheel_blocks = [WheelBlock(app=self, blocktype=config.COLLECTION),
                              WheelBlock(app=self, blocktype=config.TRAINTEST),
                              WheelBlock(app=self, blocktype=config.FEATURE),
                              WheelBlock(app=self, blocktype=config.NORMALIZER),
                              WheelBlock(app=self, blocktype=config.CLASSIFIER),
                              WheelBlock(app=self, blocktype=config.FUSION),
                              WheelBlock(app=self, blocktype=config.EVALUATION)]

        self._plugin_list = {}
        self.refresh_plugin_list()

        self._links = []

        for block in self._wheel_blocks:
            block.set_invisible()

        self._experiment_blocks = []

        # Drawing 'click to start' text
        font = tkFont.Font(family='Helvetica', size=30, weight='bold')
        self._textID = self.canvas.create_text(self.winfo_screenwidth() / 2,
                                              self.winfo_screenheight() / 2,
                                              text='Click to start',
                                              state=tk.DISABLED, font=font,
                                              justify=tk.CENTER,
                                              fill='#888888')
        
        # Recommendation
        self._recommend = tk.Button(self, text="Recommend", cursor='hand2',
                                    anchor=tk.NW, command=self.recommend,
                                    width=14)
        self._recommend.place(x=5, y=40)
        #--------------------------------------

        # Buttons, labels and text fields that appear in the top bar
        self._begin_exp_btn = tk.Button(self, text="Begin Experiment",
                                       cursor='hand2', anchor=tk.NW,
                                       command=self.generate_xml, width=14)

        self._load_exp_btn = tk.Button(self, text="Load Experiment",
                                      cursor='hand2', anchor=tk.NW,
                                      command=self.load_experiment, width=14)

        # TODO: add text fields for comments, email, tags
        # TODO: fix size of everything according to screen size
        self._top_bar_font = tkFont.Font(size=12)

        self._openset_experiment = tk.IntVar()
        self._num_iterations = tk.StringVar()
        self._experiment_name = tk.StringVar()
        self._author_name = tk.StringVar()
        self._remote_ip = tk.StringVar()
        self._remote_port = tk.StringVar()

        self._openset_experiment.set(0)
        self._num_iterations.set('1')
        self._experiment_name.set("Experiment_1")
        self._remote_ip.set(config.COMMUNICATION_HOST)
        self._remote_port.set(str(config.COMMUNICATION_PORT))

        # Bindings for clicking, releasing and moving inside the main window
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.bind('<B1-Motion>', self.on_motion)
        # Loop for checking the mouse position in order to show the top bar
        self.after(0, self.check_mouse_position)

        # Open the socket in order to begin listening for messages
        socket_communication.register_callback(self.on_message_received)
        socket_communication.begin_listening()

    def refresh_plugin_list(self):
        """Refreshes the plugin list for the application. Must be called
        manually."""
        for plugin_type in [config.FEATURE, config.CLASSIFIER,
            config.NORMALIZER, config.EVALUATION, config.TRAINTEST, config.FUSION]:
            self._plugin_list[plugin_type] = get_plugin_name_list(plugin_type,
                                                                  self._is_openset_experiment)

        self._plugin_list[config.COLLECTION] = get_collection_list()

    def check_mouse_position(self):
        """Checks the mouse position in order to (currently) show the top bar.
        Could be used for other things too. Currently does so at 33.3 fps."""
        # Register this function to be called again after 30ms
        self.after(30, self.check_mouse_position)

        # If the user is choosing a block, do nothing
        if self._wheel_visible:
            return

        # Note: need to decrease from winfo_root[x|y]() because often the
        # window isn't at (0,0) and winfo_pointerxy() returns the mouse
        # position relative to the operational system window
        mouse_x = self.winfo_pointerxy()[0] - self.winfo_rootx()
        mouse_y = self.winfo_pointerxy()[1] - self.winfo_rooty()

        # Checking mouse to display top bar
        if (mouse_x > 0 and mouse_x < self.winfo_screenwidth() and
            mouse_y < 40 and mouse_y > 0):
            self.place_top_bar()
        else:
            self.remove_top_bar()

    def place_top_bar(self):
        """Shows the top bar in the main application window."""
        # TODO: Find a way to make this screen-size independent
        if not self._top_bar_visible:
            self.canvas.create_rectangle(0, 0, self.winfo_screenwidth(), 35,
                                         fill='#333333',
                                         tags='top_bar_background')
            self._begin_exp_btn.place(x=5, y=5)
            self._load_exp_btn.place(x=155, y=5)

            # Puts label of number of executions, experiment name, author name,
            # openset
            # Remote server IP and remote server port
            self._openset_exp_check = tk.Checkbutton(self, anchor=tk.NW,
                                            variable=self._openset_experiment,
                                            command=self.switch_openset_context,
                                            text="Open-Set Experiment",
                                            width=16)

            self.canvas.create_text(570, 18, font=self._top_bar_font,
                                    fill='#FFFFFF',
                                    text="Number of Executions:",
                                    tags='top_bar_text')
            self._num_iteration_ent = tk.Entry(self, width=6,
                                              textvariable=self._num_iterations)

            self.canvas.create_text(810, 18, font=self._top_bar_font,
                                    fill='#FFFFFF',
                                    text="Experiment Name:",
                                    tags='top_bar_text')
            self._experiment_name_ent = tk.Entry(self, width=28,
                                                textvariable=self._experiment_name)

            self.canvas.create_text(1170, 18, font=self._top_bar_font,
                                    fill='#FFFFFF',
                                    text="Author:",
                                    tags='top_bar_text')
            self._author_ent = tk.Entry(self, width=13,
                                       textvariable=self._author_name)

            self.canvas.create_text(1370, 18, font=self._top_bar_font,
                                    fill='#FFFFFF',
                                    text="Server IP:",
                                    tags='top_bar_text')
            self._srv_ip_ent = tk.Entry(self, width=14,
                                       textvariable=self._remote_ip)

            self.canvas.create_text(1600, 18, font=self._top_bar_font,
                                    fill='#FFFFFF',
                                    text="Server Port:",
                                    tags='top_bar_text')
            self._srv_port_ent = tk.Entry(self, width=6,
                                         textvariable=self._remote_port)

            self._openset_exp_check.place(x=310, y=8)
            self._num_iteration_ent.place(x=670, y=8)
            self._experiment_name_ent.place(x=900, y=8)
            self._author_ent.place(x=1210, y=8)
            self._srv_ip_ent.place(x=1420, y=8)
            self._srv_port_ent.place(x=1560, y=8)

            self._top_bar_visible = True

    def switch_openset_context(self):
        """In case the user wants to perform an openset experiment, this method
           must be called in order to change collection (and eventualy classifier)
           boxes accordingly."""
        if self._openset_experiment.get() == 0:
            # Openset experiment has been turned off
            self._is_openset_experiment = False

            for block in self._experiment_blocks:
                if block.blocktype == config.COLLECTION:
                    # create new block with same collection chosen so it gets new
                    # values for openset=false
                    pass
        else:
            # Openset experiment has been turned on
            self._is_openset_experiment = True

        self.refresh_plugin_list()

    def remove_top_bar(self):
        """Removes the top bar from the main application window."""
        if self._top_bar_visible:
            self.canvas.delete('top_bar_background')
            self._openset_exp_check.place_forget()
            self._begin_exp_btn.place_forget()
            self._load_exp_btn.place_forget()
            self._num_iteration_ent.place_forget()
            self._experiment_name_ent.place_forget()
            self._author_ent.place_forget()
            self._srv_ip_ent.place_forget()
            self._srv_port_ent.place_forget()

            self.canvas.delete('top_bar_text')

            self._top_bar_visible = False

    def load_experiment(self):
        """Pops up a window asking the user for a directory, then loads the
        experiment and workflow layout of an experiment containing in that
        directory."""
        # TODO: must get the latest experiment and workflow layout
        dir_path = tkFileDialog.askdirectory(parent=self,
                                             title=('Please select the '
                                                    'directory of the '
                                                    'experiment'))

        experiment_path = os.path.join(dir_path, 'experiment.xml')
        workflow_path = os.path.join(dir_path, 'workflow.xml')

        if not os.path.exists(experiment_path):
            print("Couldn't find experiment.xml!")
            return

        if not os.path.exists(workflow_path):
            print("Couldn't find workflow.xml!")
            return
        
        # Clean design area
        if self._textID:
            self.canvas.delete(self._textID)
            self._textID = None

        # Clean current workflow
        self.clean_workflow()

        workflow_from_plugin_list(self, experiment_path,
                                                workflow_path)

    def clean_workflow(self):
        for block in self._experiment_blocks:
            block.unbind('<Enter>')
            block.unbind('<Leave>')
            block._frame.place_forget()

        self._experiment_blocks = []
        self._links = []
        self.draw_links()

        self._current_link_block = None
        self._currentMousePositionPair = None
        self._link_mode = False


    def generate_xml(self):
        """Generates an experiment and workflow xml according to the experiment
        mounted on the screen and executes the experiment."""
        if self._experiment_running:
            self.clean_experiment_feedback()

        # Checking if field for number of iterations is ok
        field_ok = True

        try:
            int(self._num_iterations.get())
        except:
            field_ok = False

        if not field_ok:
            self._num_iterations.set('1')

        #Write the openset variable into the experiment XML
        experiment_id = xmlUtils.generate_experiment_xml(self._experiment_blocks,
                                             self._links,
                                             self._num_iterations.get(),
                                             self._experiment_name.get(),
                                             self._author_name.get(),
                                             self._openset_experiment.get())

        ft = FrameworkThread(experiment_id)
        ft.start()

        self._experiment_running = True

    def on_message_received(self, msg_id, data):
        """Routine called whenever a message is received in the communication
        socket."""
        if msg_id == config.MESSAGE_MODULE_FOLDER:
            print("Module folder: %s" % data)
            # id folder
        elif msg_id == config.MESSAGE_MODULE_START:
            print("Started module %s" % data)
            # id
            module_id = int(data)

            if module_id > len(self._experiment_blocks):
                print("Module %s started is wrong!" % data)
                return

            exp_block = self._experiment_blocks[module_id - 1]
            self.canvas.create_rectangle(exp_block.x - 5, exp_block.y - 5,
                                         exp_block.x + config.BLOCK_SIZE + 4,
                                         exp_block.y + config.BLOCK_SIZE + 4,
                                         fill='#00AA00',
                                         tags='block_background_experiment',
                                         outline='')

            print("MODULE NAME: %s" % exp_block.text)
            exp_block._temporary_text = exp_block.text
            exp_block.set_text(exp_block.text + "\n0%")
        elif msg_id == config.MESSAGE_MODULE_PROGRESS:
            print("Progress: %s" % data)
            # id progress
            module_id, module_progress = data.split(" ")
            module_id = int(module_id)

            if module_id > len(self._experiment_blocks):
                print("Module %s progress is wrong!" % data)
                return

            exp_block = self._experiment_blocks[module_id - 1]
            exp_block.set_text(exp_block._temporary_text +
                               "\n%d%%" % (int(float(module_progress) * 100)))
        elif msg_id == config.MESSAGE_ITERATION_FINISH:
            print("Cleaning.")
            self.clean_experiment_feedback()
        elif msg_id == config.MESSAGE_EXPERIMENT_FINISH:
            print("Finished experiment.")
            self.clean_experiment_feedback()
            self._experiment_running = False

            # Opening results pdf
            if os.name == 'posix':
                subprocess.call(['xdg-open', data])

    def clean_experiment_feedback(self):
        """Resets all blocks in the experiment, removing all visual feedback
        from a experiment."""
        self.canvas.delete('block_background_experiment')

        print("Cleaning")
        for block in self._experiment_blocks:
            print("Block text: %s" % block._temporary_text)
            block.set_text(block._temporary_text)

        socket_communication.stop_listening()

    def remove_block(self, block):
        """Removes a block from the experiment workflow."""
        block.unbind('<Enter>')
        block.unbind('<Leave>')

        # Removing links connected to that block
        toRemove = []
        for pair in self._links:
            if block in pair:
                pairID = 'pairlink%d' % self._links.index(pair)
                self.canvas.delete(pairID)
                toRemove.append(pair)

        for pair in toRemove:
            self._links.remove(pair)

        # Update links again to undraw the removed links
        self.draw_links()

        # Logic that checks if the block removed was currently in the link
        # process and resets the application state accordingly
        if self._current_link_block == block:
            self._current_link_block = None
            self._currentMousePositionPair = None

            if self._link_mode:
                self._link_mode = False

        self._experiment_blocks.remove(block)
        block._frame.place_forget()

    def enter_link_mode(self, block):
        """Enters block link mode. In this mode, no dragging can happen and
        the next block clicked is linked with the block passed as argument to
        this method."""
        self._link_mode = True
        self._current_link_block = block
        self.after(0, self.update_temporary_link)

    def update_temporary_link(self):
        """Routine for drawing a temporary link to the mouse position."""
        if self._currentMousePositionPair:
            self._links.remove(self._currentMousePositionPair)
            self._currentMousePositionPair = None

        if not self._current_link_block:
            # Update for the last time
            self.draw_links()
            # return and break loop
            return

        mousePos = self.winfo_pointerxy()
        mousePos = (mousePos[0] - self.winfo_rootx(), mousePos[1] -
                    self.winfo_rooty())

        if (mousePos[0] >= self._current_link_block.x and mousePos[0] <=
            self._current_link_block.x + config.BLOCK_SIZE and
           mousePos[1] >= self._current_link_block.y and mousePos[1] <=
           self._current_link_block.y + config.BLOCK_SIZE):
            if self._currentMousePositionPair:
                self._links.remove(self._currentMousePositionPair)
                self._currentMousePositionPair = None

            self.draw_links()
            self.after(30, self.update_temporary_link)
            return

        # Faking a mouse "block" with x and y properties in order to make it
        # into draw_links()
        mouseBlock = namedtuple('MouseBlock', ['x', 'y'])
        mouseBlock = mouseBlock(mousePos[0], mousePos[1])

        self._currentMousePositionPair = (self._current_link_block, mouseBlock,
                                          False)

        self._links.append(self._currentMousePositionPair)

        self.draw_links()

        self.after(30, self.update_temporary_link)

    def remove_first_link(self):
        """Removes the block that was selected as first to link."""
        self._link_mode = False
        self._current_link_block = None

    def link_with(self, block):
        if not self._link_cleared:
            return

        self._links.append((self._current_link_block, block, True))
        self._link_mode = False
        self._current_link_block.exit_link()
        block.exit_link()
        self.draw_links()
        self._current_link_block = None

    def draw_links(self):
        """Draws links given tuples containing the first and second objects to
        link. The third element in the tuple indicates whether or not the link
        can be deleted via a mouse click."""
        self.canvas.delete('linkline')

        # removing bindings of links to clean everything up
        for pair in self._links:
            pairName = 'pairlink%d' % self._links.index(pair)
            self.canvas.tag_unbind(pairName, '<Enter>')
            self.canvas.tag_unbind(pairName, '<Leave>')
            self.canvas.tag_unbind(pairName, '<Button-1>')

        lineArgs = config.LINK_ATTRIBUTES

        for pair in self._links:
            pairID = self._links.index(pair)
            xDiff = pair[0].x - pair[1].x
            yDiff = pair[0].y - pair[1].y
            blockC = pair[0].get_center()
            blockS = config.BLOCK_SIZE
            blockHS = config.BLOCK_SIZE / 2
            currPos = None
            nextPos = None
            absY = abs(yDiff)
            absX = abs(xDiff)

            if pair[2]:
                lineArgs['fill'] = '#000000'
            else:
                if self._link_cleared:
                    lineArgs['fill'] = '#000000'
                else:
                    lineArgs['fill'] = '#FF0000'

            linePoints = ()

            if pair[2]:
                # dividing in cases:
                # any of the differences is lesser than the block size
                if absX < blockS:
                    absY = absY - blockS
                    # half of vertical line
                    if yDiff < 0:
                        currPos = (blockC[0], blockC[1] + blockHS)
                        nextPos = (currPos[0], currPos[1] + (absY / 2))
                    else:
                        currPos = (blockC[0], blockC[1] - blockHS)
                        nextPos = (currPos[0], currPos[1] - (absY / 2))
                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos
                    # drawing the horizontal line now
                    if xDiff < 0:
                        nextPos = (currPos[0] + absX, currPos[1])
                    else:
                        nextPos = (currPos[0] - absX, currPos[1])
                    linePoints = linePoints + nextPos
                    currPos = nextPos
                    # other half of vertical line
                    if yDiff < 0:
                        nextPos = (currPos[0], currPos[1] + (absY / 2))
                    else:
                        nextPos = (currPos[0], currPos[1] - (absY / 2))
                    linePoints = linePoints + nextPos
                elif absY < blockS:
                    absX = absX - blockS
                    # half of horizontal line
                    if xDiff < 0:
                        currPos = (blockC[0] + blockHS, blockC[1])
                        nextPos = (currPos[0] + (absX / 2), currPos[1])
                    else:
                        currPos = (blockC[0] - blockHS, blockC[1])
                        nextPos = (currPos[0] - (absX / 2), currPos[1])
                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos
                    # drawing the vertical line now
                    if yDiff < 0:
                        nextPos = (currPos[0], currPos[1] + absY)
                    else:
                        nextPos = (currPos[0], currPos[1] - absY)
                    linePoints = linePoints + nextPos
                    currPos = nextPos
                    # other half of horizontal line
                    if xDiff < 0:
                        nextPos = (currPos[0] + (absX / 2), currPos[1])
                    else:
                        nextPos = (currPos[0] - (absX / 2), currPos[1])
                    linePoints = linePoints + nextPos
                # differences are larger than the block size, divide in cases
                else:
                    absY = absY - blockHS
                    absX = absX - blockHS

                    if yDiff < 0:
                        currPos = (blockC[0], blockC[1] + blockHS)
                        nextPos = (currPos[0], currPos[1] + absY)
                    else:
                        currPos = (blockC[0], blockC[1] - blockHS)
                        nextPos = (currPos[0], currPos[1] - absY)

                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos

                    if xDiff < 0:
                        nextPos = (currPos[0] + absX, currPos[1])
                    else:
                        nextPos = (currPos[0] - absX, currPos[1])

                    linePoints = linePoints + nextPos
            else:
                # Draw link to the mouse
                if pair[0].x <= pair[1].x and absX < blockS:
                    absX = absX - blockHS

                    if yDiff < 0:
                        absY = absY - blockS
                        currPos = (blockC[0], blockC[1] + blockHS)
                        nextPos = (currPos[0], currPos[1] + (absY / 2))
                    else:
                        currPos = (blockC[0], blockC[1] - blockHS)
                        nextPos = (currPos[0], currPos[1] - (absY / 2))

                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos

                    nextPos = (currPos[0] + absX, currPos[1])
                    linePoints = linePoints + nextPos
                    currPos = nextPos

                    if yDiff < 0:
                        nextPos = (currPos[0], currPos[1] + (absY / 2) - 2)
                    else:
                        nextPos = (currPos[0], currPos[1] - (absY / 2) + 2)

                    linePoints = linePoints + nextPos
                elif pair[0].y <= pair[1].y and absY < blockS:
                    absY = absY - blockHS

                    if xDiff < 0:
                        absX = absX - blockS
                        currPos = (blockC[0] + blockHS, blockC[1])
                        nextPos = (currPos[0] + (absX / 2), currPos[1])
                    else:
                        currPos = (blockC[0] - blockHS, blockC[1])
                        nextPos = (currPos[0] - (absX / 2), currPos[1])

                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos

                    nextPos = (currPos[0], currPos[1] + absY)
                    linePoints = linePoints + nextPos
                    currPos = nextPos

                    if xDiff < 0:
                        nextPos = (currPos[0] + (absX / 2) - 2, currPos[1])
                    else:
                        nextPos = (currPos[0] - (absX / 2) + 2, currPos[1])

                    linePoints = linePoints + nextPos
                else:
                    if yDiff < 0:
                        absY = absY - blockS
                        currPos = (blockC[0], blockC[1] + blockHS)
                        nextPos = (currPos[0], currPos[1] + absY)
                    else:
                        currPos = (blockC[0], blockC[1] - blockHS)
                        nextPos = (currPos[0], currPos[1] - absY)

                    linePoints = linePoints + currPos + nextPos
                    currPos = nextPos

                    if xDiff < 0:
                        absX = absX - blockHS
                        nextPos = (currPos[0] + absX - 2, currPos[1])
                    else:
                        absX = absX + blockHS
                        nextPos = (currPos[0] - absX + 2, currPos[1])

                    linePoints = linePoints + nextPos

            # adding link id
            lineArgs['tags'] = ('linkline', 'pairlink%d' % pairID)
            self.canvas.create_line(*linePoints, **lineArgs)

        # add bindings to links of each pair with mouseover and mouseDown
        for pair in self._links:
            if len(pair) == 3 and pair[2]:
                pairName = 'pairlink%d' % self._links.index(pair)
                self.canvas.tag_bind(pairName, '<Enter>', self.on_link_over)
                self.canvas.tag_bind(pairName, '<Leave>', self.on_link_out)
                self.canvas.tag_bind(pairName, '<Button-1>', self.delete_link)

    def on_link_over(self, event):
        """Called whenever the mouse passes over a link."""
        if self._experiment_running:
            return

        linkID = self.canvas.find_closest(event.x, event.y)
        self.canvas.itemconfigure(linkID[0], width=5, fill='#FF0000')

    def on_link_out(self, event):
        """Called whenever the mouse leaves a link."""
        if self._experiment_running:
            return

        linkID = self.canvas.find_closest(event.x, event.y)
        self.canvas.itemconfigure(linkID[0], width=3, fill='#000000')

    def delete_link(self, event):
        """Called whenever the user wants to delete a link."""
        if self._experiment_running:
            return

        linkID = self.canvas.find_closest(event.x, event.y)
        linkTags = self.canvas.gettags(linkID[0])
        pairNo = int(linkTags[1][8:])  # removing the 'pairlink' beginning
        self._links.remove(self._links[pairNo])
        self.canvas.delete(linkID[0])
        self._link_deleted = True

    def on_button_press(self, event):
        """Called whenever the user clicks anywhere on the canvas."""
        # Logic for controlling the event in case the user clicks a link to
        # delete it. Since the link is also part of the canvas, this event
        # will be triggered together with delete_link(), so this logic stops
        # this function from running when this happens.
        if self._link_deleted:
            self._link_deleted = False
            return

        if self._link_mode:
            self._current_link_block.exit_link()
            self._link_mode = False
            self._current_link_block = None
            return

        if self._wheel_visible:
            self.remove_block_wheel()
        else:
            self._button_press_position['x'] = event.x
            self._button_press_position['y'] = event.y

            self.show_block_wheel()

    def on_button_release(self, event):
        """Called when the user releases the mouse button in the canvas."""
        if self._afterjob:
            self.after_cancel(self._afterjob)

    def on_motion(self, event):
        """Called when the user moves the mouse wihin the canvas."""
        if self._afterjob:
            self.after_cancel(self._afterjob)
            self._afterjob = None

    def show_block_wheel(self):
        """Shows the block wheel for the user to choose a block to add in the
        experiment."""
        # If first time showing the wheel, simply remove the background guide
        # text
        if self._textID:
            self.canvas.delete(self._textID)
            self._textID = None

        self._afterjob = None
        self._wheel_visible = True

        # Gray out everything else
        for block in self._experiment_blocks:
            block.gray_out()
            if block._context_visible:
                block._context_box.set_invisible()
                block._context_visible = False

        self.canvas.configure(bg='#444444')

        # Display the wheel
        total_blocks = len(config.APPLICATION_BLOCKS)
        for i in xrange(total_blocks):
            teta = math.radians(i * (float(360) / total_blocks))
            self._wheel_blocks[i].set_visible()
            self._wheel_blocks[i].position(self._button_press_position['x'] -
                                           (config.BLOCK_SIZE / 2) +
                                           (config.WHEEL_RADIUS *
                                            math.sin(teta)),
                                           self._button_press_position['y'] -
                                           (config.BLOCK_SIZE / 2) -
                                           (config.WHEEL_RADIUS *
                                            math.cos(teta)))
            tk.Misc.lift(self._wheel_blocks[i]._frame)

    def remove_block_wheel(self):
        """Removes the block wheel in the main application window."""
        if self._wheel_visible:
            for block in self._wheel_blocks:
                block.set_invisible()

            for block in self._experiment_blocks:
                block.remove_gray()

            self.canvas.configure(bg='#AAAAAA')

            self._wheel_visible = False

    def add_block_on_mouse_position(self, blocktype):
        """Adds a block of the type blocktype centered in the last mouse
        position registered when the user clicked on the canvas."""
        self.remove_block_wheel()

        block = ExperimentBlock(app=self, blocktype=blocktype,
                                x=self._button_press_position['x'] -
                                (config.BLOCK_SIZE / 2),
                                y=self._button_press_position['y'] -
                                (config.BLOCK_SIZE / 2))
        block.bind('<Enter>', self.on_block_over)
        block.bind('<Leave>', self.on_block_out)
        self._experiment_blocks.append(block)

    def add_block(self, block):
        block.bind('<Enter>', self.on_block_over)
        block.bind('<Leave>', self.on_block_out)
        self._experiment_blocks.append(block)

    def register_links(self, links):
        self._links = links
        self.draw_links()

    def on_block_over(self, event):
        """Called when the mouse passes over a block."""
        if self._link_mode:
            # Check whether block with mouse over accepts the block to connect
            # type and set a flag that it is possible to link them
            connect_blocktypes = event.widget._context_box._able_to_link

            if connect_blocktypes is not None:
                self._link_cleared = (self._current_link_block.blocktype in
                                      connect_blocktypes)
            else:
                # For blocks that don't say what can connect to them, just
                # connect everything
                self._link_cleared = True

    def on_block_out(self, event):
        """Called when the mouse leaves a block."""
        if self._link_mode:
            self._link_cleared = True
    
    # Recommend
    def recommend(self):
        """"""
        
        self._recommend_dict = {}
        
        def read_experiment_file(path_file):
            """
            Read the XML file and generates the sequence
            """
            
            import xml.etree.cElementTree as ET
                      
            COLLECTION = 0
            TRAIN_TEST = 1
            DESCRIPTOR = 2
            NORMALIZER = 3
            CLASSIFIER = 4

            sequence = []
            
            tree = ET.parse(path_file)
            root = tree.getroot()
            for child in root:
                if child.tag == 'collection':
                    sequence.insert(COLLECTION, child.attrib['name'])
                elif child.tag == 'train_test_method':
                    sequence.insert(TRAIN_TEST, child.attrib['name'])
                elif child.tag == 'descriptor':
                    sequence.insert(DESCRIPTOR, child.attrib['name'])
                elif child.tag == 'normalizer':
                    sequence.insert(NORMALIZER, child.attrib['name'])
                elif child.tag == 'classifier':
                    sequence.insert(CLASSIFIER, child.attrib['name'])
            
            return sequence
        
        DIRPATH = 0
        FILENAMES = 2
        ZERO_INDEX = 0
        
        #Train set
        train_dict = {}
        experiments_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
        for walk_tuple in os.walk(experiments_path):
            if 'experiment.xml' in walk_tuple[FILENAMES]:
                train_dict[walk_tuple[DIRPATH].split(os.sep)[-1]] = read_experiment_file(os.path.join(walk_tuple[DIRPATH], 'experiment.xml'))
        
        #Test
        experiment_id = xmlUtils.generate_experiment_xml(self._experiment_blocks,
                                             self._links,
                                             self._num_iterations.get(),
                                             self._experiment_name.get(),
                                             self._author_name.get(),
                                             self._openset_experiment.get())
        test_sequence = read_experiment_file(os.path.join(experiments_path, "Experiment_" + experiment_id, "experiment.xml"))
        
        #Remove XML and folder created for the test
        subprocess.call(['rm', '-r', os.path.join(experiments_path, "Experiment_" + experiment_id)])
        
        #Create recommender window
        recommend_window = tk.Toplevel(self)
        recommend_window.wm_title("Recommender")
        
        #Options for the recommender
        frame_radio = tk.Frame(recommend_window)
        recommend_radio = tk.StringVar()
        recommendation_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recommendation'))
        recommend_radio_list = []
        for recommendation_plugin_folder in os.walk(recommendation_folder).next()[1]:
            sys.path.append(os.path.join(recommendation_folder, recommendation_plugin_folder))
            
            recommendation_plugin = __import__('plugin_{0}'.format(recommendation_plugin_folder))
            recommend_radio_item = tk.Radiobutton(frame_radio, text=recommendation_plugin.get_text(), value=recommendation_plugin_folder, variable=recommend_radio)
            recommend_radio_item.pack(anchor=tk.W)
            recommend_radio_list.append(recommend_radio_item)
            
            sys.path.remove(os.path.join(recommendation_folder, recommendation_plugin_folder))
            
        frame_radio.pack()
        frame_radio.grid(row=ROW_RADIO, column=COLUMN_RADIO_BUTTON, rowspan=RECOMMENDATION_SHOW-1)
        
        #Frame to present the recommended workflows
        recommend_frame = tk.Frame(recommend_window)
        recommend_frame.pack()
        recommend_frame.grid(row=ROW_RADIO, column=COLUMN_CHECKBUTTON, rowspan=RECOMMENDATION_SHOW)

        #Recommender button
        recommend_button = tk.Button(recommend_window, text="Recommend", cursor='hand2', anchor=tk.S, command=lambda: self.recom_main(train_dict, test_sequence, recommend_radio.get(), recommend_frame), width=14)
        recommend_button.pack()
        recommend_button.grid(row=ROW_BUTTON, column=COLUMN_RADIO_BUTTON)

    def recom_main(self, train_dict, test_sequence, recommend_radio_val, recommend_window):
        # Receives a list of strings, where each string is the name of the folder of the recommended experiment
        recom = recommendation.main(train_dict, test_sequence, recommend_radio_val, RECOMMENDATION_SHOW)

        # Getting image of past experiments
        for index, item in enumerate(recom):
            try:
                self._recommend_dict[index].grid_forget()
            except:
                pass
            curr_img = Image.open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", item[0], "graph.png")))
            curr_img_res = curr_img.resize((int(0.8*self.winfo_screenwidth()),curr_img.size[1]), Image.ANTIALIAS)
            curr_img_tk = ImageTk.PhotoImage(curr_img_res)
            curr_button = tk.Button(recommend_window, image=curr_img_tk, cursor='hand2', anchor=tk.E)
            self._recommend_dict[index] = curr_button
            self._recommend_dict[index].experiment_folder = copy.deepcopy(item[0])
            self._recommend_dict[index].image = curr_img_tk
            self._recommend_dict[index].config(command=lambda folder=self._recommend_dict[index].experiment_folder: self.populate_experiment(folder))
            self._recommend_dict[index].grid(row=index, column=COLUMN_RECOMMENDATION)

    def populate_experiment(self, experiment):
        if self._textID:
            self.canvas.delete(self._textID)
            self._textID = None
        self.clean_workflow()
        experiment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", experiment, 'experiment.xml'))
        workflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", experiment, 'workflow.xml'))
        workflow_from_plugin_list(self, experiment_path,
                                                workflow_path)

    def generate_log(self, query, recommendations, checkbutton_variable, relevance_window):
        """
        """
        
        recommendation_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recommendation results"))
        relevance_path = os.path.join(recommendation_folder, "relevance_" + query + ".txt")
        
        relevance_file = open(relevance_path, "w")
        for recom in recommendations:
            relevance_file.write(query + "\t" + recom[0] + "\t")
            for recommend_item in ['jaccard', 'sorensen', 'jaro_winkler', 'tfidf']:
                relevance_file.write("w[" + recommend_item + "]=" + str(recom[1][recommend_item]) + "\t")
            relevance_file.write(str(checkbutton_variable[recom[0]].get()) + "\n")
        relevance_file.close()
        
        relevance_window.destroy()
