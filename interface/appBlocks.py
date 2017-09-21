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
from tkFileDialog import askopenfilename
import tkFont
import ttk

import os
import sys

import config
from appUtils import *
import validation_tests
import xmlUtils
import time

tags_added = {}
tags_child = {}

class Block(tk.Canvas):
    """Base building block for the interface."""

    def __init__(self, app, size, **kwargs):
        self._frame = tk.Frame(app)

        tk.Canvas.__init__(self, self._frame, width=size[0], height=size[1],
                           highlightthickness=0)

        self.pack(side=tk.LEFT)

        # : Size of this block
        self.size = size

        # : Points to the application root
        self.app = app

        # First check if there are two colors defined. Then create gradient
        # When drawing colors, use a 1 pixel border to later draw the outline
        if 'color1' in kwargs and 'color2' in kwargs:
            self._color1 = kwargs['color1']
            self._color2 = kwargs['color2']
        else:
            if 'color1' in kwargs:
                self._color1 = kwargs['color1']
            elif 'color' in kwargs:
                self._color1 = kwargs['color']
            else:
                self._color1 = '#000000'

        self._bg_extra_size_y = 0
        self.draw_background()

        if 'x' in kwargs:
            self.x = kwargs['x']
        else:
            self.x = 0

        if 'y' in kwargs:
            self.y = kwargs['y']
        else:
            self.y = 0

        if 'in_' in kwargs:
            self._frame.place(x=self.x, y=self.y, in_=kwargs['in_'])
        else:
            self._frame.place(x=self.x, y=self.y)

    def gray_out(self):
        """Adds gray to all colors in the block to give the effect of a
        transparent black in front of the block.
        Also changes the block's cursor to the default arrow cursor."""
        # Clear background
        self.delete('background')
        self._r1, self._g1, self._b1 = color_string_subtract(self._color1,
                                                             '#AAAAAA', False)
        self._color1 = convert_RGB_to_color_string(clamp(self._r1, 0, 255),
                                                   clamp(self._g1, 0, 255),
                                                   clamp(self._b1, 0, 255))
        if hasattr(self, '_color2'):
            self._r2, self._g2, self._b2 = color_string_subtract(self._color2,
                                                                 '#AAAAAA',
                                                                 False)
            self._color2 = convert_RGB_to_color_string(clamp(self._r2, 0, 255),
                                                       clamp(self._g2, 0, 255),
                                                       clamp(self._b2, 0, 255))

        self.draw_background()

    def remove_gray(self):
        """Removes the "transparent black" effect from the block. Also changes
        the block's cursor back to the hand cursor."""
        # Clear background
        self.delete('background')
        self._r1, self._g1, self._b1 = (self._r1 + 170, self._g1 + 170,
                                        self._b1 + 170)
        self._color1 = convert_RGB_to_color_string(clamp(self._r1, 0, 255),
                                                   clamp(self._g1, 0, 255),
                                                   clamp(self._b1, 0, 255))
        if hasattr(self, '_color2'):
            self._r2, self._g2, self._b2 = (self._r2 + 170, self._g2 + 170,
                                            self._b2 + 170)
            self._color2 = convert_RGB_to_color_string(clamp(self._r2, 0, 255),
                                                       clamp(self._g2, 0, 255),
                                                       clamp(self._b2, 0, 255))

        self.draw_background()

    def create_border(self):
        """Creates a black border outline in the block. Do not call it
        directly."""
        # Black border around the block
        self.create_rectangle(0, 0, self.size[0] - 1,
                              self.size[1] - 1 + self._bg_extra_size_y,
                              fill='', outline='#000000', tags='background')

    def create_gradient(self, sx, sy, ex, ey, color1, color2):
        """Creates a gradient inside the block by drawing lines and
        interpolating both color1 and color2 linearly.
        Currently only supports vertical gradient starting at color1 and
        ending at color2.
        sx and sy stand for start x and start y.
        ex and ey stand for end x and end y."""
        # First of all, convert colors to correspondent RGB integers
        a1, r1, g1, b1 = convert_color_string_to_RGB(color1)
        a2, r2, g2, b2 = convert_color_string_to_RGB(color2)

        # Amount of steps to go through
        ySteps = ey - sy

        step = 0

        # Main drawing loop
        while step < ySteps:
            percentage = float(step) / float(ySteps)

            r = ((1 - percentage) * r1) + (percentage * r2)
            g = ((1 - percentage) * g1) + (percentage * g2)
            b = ((1 - percentage) * b1) + (percentage * b2)

            color = convert_RGB_to_color_string(int(r), int(g), int(b))

            self.create_line(sx, sy + step, ex, sy + step, fill=color,
                             tags='background')

            step = step + 1

    def draw_background(self):
        """Draws the background of the block. The background is made of:
        gradient, border and text."""
        self.delete('background')

        if hasattr(self, '_color2'):
            self.create_gradient(1, 1, self.size[0] - 1,
                                 self.size[1] - 1 + self._bg_extra_size_y,
                                 color1=self._color1, color2=self._color2)
        else:
            self.create_rectangle(1, 1, self.size[0] - 1,
                                  self.size[1] - 1 + self._bg_extra_size_y,
                                  fill=self._color1, outline='',
                                  tags='background')

        self.create_border()
        self.tag_lower('background')

    def position(self, x, y):
        """Places the block in the position specified in the arguments."""
        self.x = x
        self.y = y

        self._frame.place_configure(x=self.x, y=self.y)

    def move(self, dx, dy):
        """Moves the block by the amount specified in the arguments."""
        self.x += dx
        self.y += dy

        self._frame.place_configure(x=self.x, y=self.y)

    def get_center(self):
        """Returns a tuple with the x and y positions of the center of the
        block."""
        return (self.x + (self.size[0] / 2), self.y + (self.size[1] / 2))

    def set_visible(self):
        """Makes the block visible within the application."""
        self._frame.place(x=self.x, y=self.y)

    def set_invisible(self):
        """Hides the block from the application."""
        self._frame.place_forget()


class ContextBlock(Block):
    """Block for each AppBlock's context."""
    CONTEXT_SIZE = (280, 400)

    ITEM_PADDING = 10
    LABEL_PADDING = 5

    def __init__(self, app, **kwargs):
        self._items = []
        self.parameters = {}
        self.parameter_types = {}
        self.title = ""
        self.plugin_name = ""
        self._next_pos = ContextBlock.LABEL_PADDING
        self._attached_block = None
        self._attached_block_previous_text = None
        self._able_to_link = []

        self._title_font = tkFont.Font(family='Helvetica', size=14,
                                       weight='bold')
        self._label_font = tkFont.Font(family='Helvetica', size=9,
                                       weight='bold')

        Block.__init__(self, app, ContextBlock.CONTEXT_SIZE, **kwargs)

    def get_parameters(self):
        parameters = {}

        for prm in self.parameters:
            if prm in self.parameter_types:
                if self.parameter_types[prm] == "string":
                    parameters[prm] = str(self.parameters[prm].get())
                elif self.parameter_types[prm] == "integer":
                    parameters[prm] = int(self.parameters[prm].get())
                elif self.parameter_types[prm] == "float":
                    parameters[prm] = float(self.parameters[prm].get())
                elif self.parameter_types[prm] == "checkbox":
                    if self.parameters[prm].get() == 1:
                        parameters[prm] = True
                    else:
                        parameters[prm] = False
                elif self.parameter_types[prm] == "list":
                    parameters[prm] = str(self.parameters[prm].get())
                elif self.parameter_types[prm] == "range":
                    parameters[prm] = (str(self.parameters[prm][0].get()) +
                        "," + str(self.parameters[prm][1].get()) + "," +
                        str(self.parameters[prm][2].get()))
                elif self.parameter_types[prm] == "filedialog":
                    parameters[prm] = str(self.parameters[prm].get())
            else:
                parameters[prm] = str(self.parameters[prm].get())

        return parameters

    def get_parameter_list(self):
        return self.parameters, self.parameter_types

    def update_pos(self, value):
        self._next_pos = self._next_pos + value + ContextBlock.ITEM_PADDING
        self._bg_extra_size_y = max(0, self._next_pos -
                                    ContextBlock.CONTEXT_SIZE[1])
        self.draw_background()
        
        self.configure(scrollregion=(0, 0, ContextBlock.CONTEXT_SIZE[0],
                                         self._next_pos))

    def attach(self, block):
        xPos = block.x + config.BLOCK_SIZE - 1
        yPos = block.y - ((ContextBlock.CONTEXT_SIZE[1] -
                           config.BLOCK_SIZE) / 2)
        self.position(xPos, yPos)
        self._attached_block = block

    def populate_buttons(self, name_list):
        """Populates the context box with buttons with names of the
        plugins that when click activate the context of the selected plugin."""
        def update_scroll(event):
            self.configure(scrollregion=self.bbox("all"))
        
        self.b = tk.Scrollbar(self._frame, command=self.yview, cursor='hand2', orient=tk.VERTICAL)
#        scrollbar = tk.Scrollbar(relevance_window, orient=tk.VERTICAL, command=temp_canvas.yview)
        self.b.pack(side=tk.RIGHT, fill=tk.Y)
        self.configure(yscrollcommand=self.b.set)
        self.bind("<Configure>", update_scroll)

        self._next_pos = ContextBlock.LABEL_PADDING

        for name in name_list:
            btn = tk.Button(self, text=name[0], cursor='hand2', anchor=tk.NW,
                            command=self.button_click_callback(name[1]),
                            width=16, justify=tk.RIGHT, font=self._label_font)

            remaining_width = (ContextBlock.CONTEXT_SIZE[0] -
                               btn.winfo_reqwidth())

            self.create_window(remaining_width / 2, self._next_pos,
                               anchor=tk.NW, width=btn.winfo_reqwidth(),
                               height=btn.winfo_reqheight(),
                               window=btn, tags='plugin-button')

            self.update_pos(btn.winfo_reqheight())

    def button_click_callback(self, plugin_name):
        return lambda: self.switch_context(plugin_name)

    def switch_context(self, plugin_name):
        self.delete('plugin-button')

        self._next_pos = ContextBlock.LABEL_PADDING
        self._bg_extra_size_y = max(0, self._next_pos -
                                    ContextBlock.CONTEXT_SIZE[1])
        self.configure(scrollregion=(0, 0, ContextBlock.CONTEXT_SIZE[0],
                                         self._next_pos))
        self.draw_background()

        self._items = []
        self._attached_block_previous_text = self._attached_block.text

        if self._attached_block.blocktype == config.COLLECTION:
            self._attached_block.set_text(xmlUtils.read_collection_name(
                plugin_name))
        else:
            self._attached_block.set_text(xmlUtils.read_plugin_name(
                plugin_name))

        self._attached_block._temporary_text = self._attached_block.text

        self.plugin_name = plugin_name
        # Tell xmlUtils to read plugin's xml and populate it
        if self._attached_block.blocktype == config.COLLECTION:
            xmlUtils.read_collection_description(plugin_name, self)
        else:
            xmlUtils.read_plugin_description(plugin_name, self)

    def set_able_to_link(self, block_type):
        self._able_to_link.append(block_type)

    def add_title(self, title):
        """Adds a title in the context box."""
        self.title = title

        xPos = ContextBlock.CONTEXT_SIZE[0] / 2

        textID = self.create_text(xPos, self._next_pos, text=title,
                                  width=ContextBlock.CONTEXT_SIZE[0],
                                  state=tk.DISABLED, font=self._title_font,
                                  justify=tk.CENTER, anchor=tk.N)

        bbox = self.bbox(textID)

        self.update_pos(bbox[3] - bbox[1])

    def add_label(self, name, tags=None):
        """Adds a label in the context box. Does not update the next position.
        Returns the next suitable width position in the current row."""
        if not tags == None:
            name = "{0}_{1}".format(tags, name)
        textID2 = self.create_text(ContextBlock.LABEL_PADDING, self._next_pos,
                                   text=name,
                                   width=ContextBlock.CONTEXT_SIZE[0],
                                   state=tk.DISABLED, font=self._label_font,
                                   justify=tk.LEFT, anchor=tk.NW, tags=tags)

        bbox = self.bbox(textID2)

        return (ContextBlock.LABEL_PADDING + bbox[2] - bbox[0] +
                ContextBlock.LABEL_PADDING)

    def add_label_with_linebreak(self, name):
        """Adds a label in the context box and updates the next position."""
        textID2 = self.create_text(ContextBlock.LABEL_PADDING, self._next_pos,
                                   text=name,
                                   width=ContextBlock.CONTEXT_SIZE[0],
                                   state=tk.DISABLED, font=self._label_font,
                                   justify=tk.LEFT, anchor=tk.NW)

        bbox = self.bbox(textID2)

        self.update_pos(bbox[3] - bbox[1])

    def add_dropdown(self, name, items, default=None, fixed=False, reference=None, tags='plugin-item'):
        """Adds a dropdown in the context box."""
        stringVar = tk.StringVar()
        stringVar.old_text = ""

        validationWrapper = self.register(
            validation_tests.validate_combobox(items, stringVar))

        invalidWrapper = self.register(
            validation_tests.return_previous_text(stringVar))

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currDDWidth = 20
        dropdown = ttk.Combobox(self, width=currDDWidth, height=5,
                                values=items,
                                validatecommand=(validationWrapper, '%P'),
                                validate='focusout',
                                invalidcommand=invalidWrapper,
                                textvariable=stringVar, state=widgetstate)

        def handler_update_plugin_description(event):
            if reference is not None:
                current = items[dropdown.current()]
                reference_path = "{0}{3}{1}{3}{2}.xml".format(os.getcwd(),
                        reference, current, os.sep)
                
                try:
                    if name in tags_added[tags]:
                        name_tag = name.replace(' ', '_')
                        self.delete(name_tag)
                        if name in tags_child.keys():
                            for child in tags_child[name]:
                                del self.parameters[child]
                            del tags_child[name]
                        tags_added[tags].remove(name)
                except:
                    pass

                try:
                    child_list = xmlUtils.update_plugin_description(reference_path, name, self)
                    tags_child[name] = child_list
                except:
                    pass
                
                if tags not in tags_added:
                    tags_added[tags] = []
                tags_added[tags].append(name)
        
        # Bind dropdown selected item
        dropdown.bind('<<ComboboxSelected>>', handler_update_plugin_description)
        #

        # While dropdown's width is larger than the context box can hold,
        # reduce its width. We add LABEL_PADDING here for aesthetic purposes
        # on the dropdown's "tail"
        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               dropdown.winfo_reqwidth() + labelWidth):
            currDDWidth -= 1
            dropdown.configure(width=currDDWidth)

        if default is not None:
            if default in items:
                dropdown.current(items.index(default))

        self._items.append(dropdown)

        # Return to create window
        self.create_window(labelWidth, self._next_pos,
                anchor=tk.NW, width=dropdown.winfo_reqwidth(),
                height=dropdown.winfo_reqheight(),
                window=dropdown, tags=tags)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'string'

        # This line needs to be here in order to update the widget's height
        dropdown.update_idletasks()
        self.update_pos(dropdown.winfo_reqheight())

    def add_integer(self, name, default=None, fixed=False, tags='plugin-item', reference=None):
        """Adds an entry field that only allows integers to be entered."""
        stringVar = tk.StringVar()

        if default is None:
            stringVar.old_text = ""
        else:
            stringVar.set(default)
            stringVar.old_text = default

        validationWrapper = self.register(
            validation_tests.validate_integer(stringVar))

        invalidWrapper = self.register(
            validation_tests.return_previous_text(stringVar))

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currETWidth = 25
        entry = tk.Entry(self, width=currETWidth,
                         validatecommand=(validationWrapper, '%P'),
                         validate='focusout',
                         invalidcommand=invalidWrapper,
                         textvariable=stringVar, state=widgetstate)
        
        def handler_update_plugin_description(event):
            if reference is not None and entry.get() is not '':
                entry_value = entry.get()
                reference_path = "{0}{1}{2}".format(os.getcwd(), os.sep,
                        reference)
                
                try:
                    if name in tags_added[tags]:
                        name_tag = name.replace(' ', '_')
                        self.delete(name_tag)
                        if name in tags_child.keys():
                            for child in tags_child[name]:
                                del self.parameters[child]
                            del tags_child[name]
                        tags_added[tags].remove(name)
                except:
                    pass
                
                try:
                    if reference[-3:] == ".py":
                        sys.path.append(os.path.dirname(reference_path))
                        software = __import__(reference_path.split(os.sep)[-1][:-3])
                        reference_path = software.main(entry.get())
                    child_list = xmlUtils.update_plugin_description(reference_path, name, self)
                    tags_child[name] = child_list
                except:
                    pass
                
                if tags not in tags_added:
                    tags_added[tags] = []
                tags_added[tags].append(name)
        
        # Bind selection
        entry.bind('<FocusOut>', handler_update_plugin_description)
        #

        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               entry.winfo_reqwidth() + labelWidth) and currETWidth > 3:
            currETWidth -= 1
            entry.configure(width=currETWidth)

        self._items.append(entry)

        #Create window to fix the entry in the block
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry.winfo_reqwidth(), height=entry.winfo_reqheight(),
                window=entry, tags=tags)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'integer'

        entry.update_idletasks()
        
        #Changed the winfo to the correct function
        self.update_pos(entry.winfo_reqheight())

    def add_float(self, name, default=None, fixed=False, tags='plugin-item'):
        """Adds an entry field that only allows floats to be entered."""
        stringVar = tk.StringVar()

        if default is None:
            stringVar.old_text = ""
        else:
            stringVar.set(default)
            stringVar.old_text = default

        validationWrapper = self.register(
            validation_tests.validate_float(stringVar))

        invalidWrapper = self.register(
            validation_tests.return_previous_text(stringVar))

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currETWidth = 20
        entry = tk.Entry(self, width=currETWidth,
                         validatecommand=(validationWrapper, '%P'),
                         validate='focusout',
                         invalidcommand=invalidWrapper,
                         textvariable=stringVar, state=widgetstate)

        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               entry.winfo_reqwidth() + labelWidth):
            currETWidth -= 1
            entry.configure(width=currETWidth)

        self._items.append(entry)

        # Create window
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry.winfo_reqwidth(), height=entry.winfo_reqheight(),
                window=entry, tags=tags)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'float'

        entry.update_idletasks()
        self.update_pos(entry.winfo_reqheight())

    def add_checkbox(self, name, default=None, fixed=False, tags='plugin-item'):
        """Adds a checkbox with a label."""
        intVar = tk.IntVar()
        intVar.set(0)

        if default is not None:
            if default.lower() == "true":
                intVar.set(1)
            else:
                intVar.set(0)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        check = tk.Checkbutton(self, anchor=tk.NW, text=name, width=13,
                               variable=intVar, state=widgetstate)

        self._items.append(check)
        
        #Create window to fix the check in the block
        self.create_window(ContextBlock.LABEL_PADDING, self._next_pos,
                anchor=tk.NW, width=check.winfo_reqwidth()+30,
                height=check.winfo_reqheight(), window=check, tags=tags)

        self.parameters[name] = intVar
        self.parameter_types[name] = 'checkbox'

        check.update_idletasks()
        
        #Changed the winfo to the correct function
        self.update_pos(check.winfo_reqheight())

    def add_list(self, name, default=None, fixed=False, tags='plugin-item'):
        """Adds a list for the user to input comma-separated values."""
        stringVar = tk.StringVar()

        if default:
            stringVar.set(default)

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currETWidth = 20
        entry = tk.Entry(self, width=currETWidth, state=widgetstate, textvariable=stringVar)

        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               entry.winfo_reqwidth() + labelWidth):
            currETWidth -= 1
            entry.configure(width=currETWidth)

        self._items.append(entry)

        # Create window
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry.winfo_reqwidth(), height=entry.winfo_reqheight(),
                window=entry, tags=tags)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'list'

        entry.update_idletasks()
        self.update_pos(entry.winfo_reqheight())

    def add_range(self, name, start, stop, step, default_start, default_stop,
                  default_step, fixed=False, tags='plugin-item'):
        """Adds three entries for the user to input start, stop and step values
        for a range."""
        stringVar_start = tk.StringVar()
        stringVar_stop = tk.StringVar()
        stringVar_step = tk.StringVar()

        if start == 'integer':
            validationWrapper = self.register(
            validation_tests.validate_integer(stringVar_start))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_start))
        else:
            validationWrapper = self.register(
            validation_tests.validate_float(stringVar_start))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_start))

        if stop == 'integer':
            validationWrapper = self.register(
            validation_tests.validate_integer(stringVar_stop))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_stop))
        else:
            validationWrapper = self.register(
            validation_tests.validate_float(stringVar_stop))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_stop))

        if step == 'integer':
            validationWrapper = self.register(
            validation_tests.validate_integer(stringVar_step))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_step))
        else:
            validationWrapper = self.register(
            validation_tests.validate_float(stringVar_step))
            invalidWrapper = self.register(
                validation_tests.return_previous_text(stringVar_step))

        if default_start:
            stringVar_start.set(default_start)
        if default_stop:
            stringVar_stop.set(default_stop)
        if default_step:
            stringVar_step.set(default_step)

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        entry_start = tk.Entry(self, width=4, state=widgetstate, textvariable=stringVar_start)
        size_start = entry_start.winfo_reqwidth()
        entry_stop = tk.Entry(self, width=4, state=widgetstate, textvariable=stringVar_stop)
        size_stop = entry_stop.winfo_reqwidth()
        entry_step = tk.Entry(self, width=4, state=widgetstate, textvariable=stringVar_step)

        self._items.append(entry_start)
        self._items.append(entry_stop)
        self._items.append(entry_step)

        # Create window
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry_start.winfo_reqwidth(), height=entry_start.winfo_reqheight(),
                window=entry_start, tags=tags)
        self.create_window(labelWidth + size_start + 5, self._next_pos, anchor=tk.NW,
                width=entry_stop.winfo_reqwidth(), height=entry_stop.winfo_reqheight(),
                window=entry_stop, tags=tags)
        self.create_window(labelWidth + size_start + size_stop + 10, self._next_pos, anchor=tk.NW,
                width=entry_step.winfo_reqwidth(), height=entry_step.winfo_reqheight(),
                window=entry_step, tags=tags)

        self.parameters[name] = (stringVar_start, stringVar_stop, stringVar_step)
        self.parameter_types[name] = 'range'

        entry_start.update_idletasks()
        entry_stop.update_idletasks()
        entry_step.update_idletasks()
        self.update_pos(entry_start.winfo_reqheight())

    def add_filedialog(self, name, default_path, fixed=False, tags='plugin-item'):
        """Adds a file dialog field with a browse button to choose a path."""
        stringVar = tk.StringVar()

        if default_path is None:
            stringVar.old_text = ""
        else:
            stringVar.set(default_path)
            stringVar.old_text = default_path

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currETWidth = 5
        entry = tk.Entry(self, width=currETWidth,
                         textvariable=stringVar, state=widgetstate)

        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               entry.winfo_reqwidth() + labelWidth) and currETWidth > 3:
            currETWidth -= 1
            entry.configure(width=currETWidth)

        self._items.append(entry)

        #Create window to fix the entry in the block
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry.winfo_reqwidth(), height=entry.winfo_reqheight(),
                window=entry, tags=tags)

        btn = tk.Button(self, text="Browse", cursor='hand2', anchor=tk.NW,
                            command=lambda: stringVar.set(askopenfilename(filetypes=[("All files", ".*")])),
                            width=5, justify=tk.RIGHT, font=self._label_font)

        self.create_window(labelWidth + entry.winfo_reqwidth(), self._next_pos, anchor=tk.NW,
                width=btn.winfo_reqwidth(), height=btn.winfo_reqheight(),
                window=btn, tags=tags)
        
        self._items.append(btn)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'filedialog'

        entry.update_idletasks()
        
        #Changed the winfo to the correct function
        self.update_pos(entry.winfo_reqheight())
    
    def add_string(self, name, default=None, fixed=False, tags='plugin-item', reference=None):
        """Adds an entry field that allows strings to be entered."""
        stringVar = tk.StringVar()

        if default is None:
            stringVar.old_text = ""
        else:
            stringVar.set(default)
            stringVar.old_text = default

        labelWidth = self.add_label(name, tags=tags)

        if fixed:
            widgetstate = tk.DISABLED
        else:
            widgetstate = tk.NORMAL

        currETWidth = 25
        entry = tk.Entry(self, width=currETWidth,
                         textvariable=stringVar, state=widgetstate)

        while (ContextBlock.CONTEXT_SIZE[0] < ContextBlock.LABEL_PADDING +
               entry.winfo_reqwidth() + labelWidth) and currETWidth > 3:
            currETWidth -= 1
            entry.configure(width=currETWidth)

        self._items.append(entry)

        #Create window to fix the entry in the block
        self.create_window(labelWidth, self._next_pos, anchor=tk.NW,
                width=entry.winfo_reqwidth(), height=entry.winfo_reqheight(),
                window=entry, tags=tags)

        self.parameters[name] = stringVar
        self.parameter_types[name] = 'string'

        entry.update_idletasks()
        
        #Changed the winfo to the correct function
        self.update_pos(entry.winfo_reqheight())

    def set_visible(self):
        Block.set_visible(self)
        tk.Misc.lift(self._frame)


class AppBlock(Block):
    """Block representing each structure."""

    # : Size of block in the screen, in pixels.

    def __init__(self, app, **kwargs):
        size = (config.BLOCK_SIZE, config.BLOCK_SIZE)

        Block.__init__(self, app, size, **kwargs)
        self.configure(cursor='hand2')

        if 'text' in kwargs:
            self.text = kwargs['text']
            self.textID = self.draw_text()

    def set_text(self, text):
        """Changes the text of the current block."""
        self.text = text
        self.delete('block-text')
        self.textID = self.draw_text()

    def draw_text(self):
        """Draws the text of the block in the middle of it."""
        font = tkFont.Font(family='Helvetica', size=9, weight='bold')

        textID = self.create_text(config.BLOCK_SIZE / 2,
                                  config.BLOCK_SIZE / 2, text=self.text,
                                  width=config.BLOCK_SIZE, state=tk.DISABLED,
                                  font=font, justify=tk.CENTER,
                                  tags='block-text')
        return textID

    def gray_out(self):
        Block.gray_out(self)

        # Raise text to the top again
        self.lift(self.textID)

        # Switch to common cursor
        self.configure(cursor='arrow')

    def remove_gray(self):
        Block.remove_gray(self)

        # Raise text to the top again
        self.lift(self.textID)

        # Use hand cursor again
        self.configure(cursor='hand2')


class WheelBlock(AppBlock):
    """Buttons that will be displayed as blocks in the selection wheel."""

    def __init__(self, app, blocktype, **kwargs):
        if blocktype in config.BLOCK_TYPE_STYLES:
            kwargs.update(config.BLOCK_TYPE_STYLES[blocktype])
        else:
            raise ValueError("invalid block type argument '%s'." % blocktype)

        AppBlock.__init__(self, app, **kwargs)

        self._clicked = False
        self._blocktype = blocktype
        self._app = app
        self._highlightrectangle = None

        self.bind('<ButtonPress-1>', self.on_mouse_down)
        self.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.bind('<Enter>', self.on_mouse_over)
        self.bind('<Leave>', self.on_mouse_out)

    def on_mouse_down(self, event):
        self._clicked = True

    def on_mouse_up(self, event):
        if self._clicked:
            # Add new block of the same type in the mouse position
            self._app.add_block_on_mouse_position(self._blocktype)

        self._clicked = False

    def on_mouse_over(self, event):
        self._highlightrectangle = self._app.canvas.create_rectangle(
            self.x - 1, self.y - 1, self.x + config.BLOCK_SIZE,
            self.y + config.BLOCK_SIZE, fill='#000000')

    def on_mouse_out(self, event):
        if self._highlightrectangle:
            self._app.canvas.delete(self._highlightrectangle)


class ExperimentBlock(AppBlock):
    """Represents all final blocks that the user can create."""

    def __init__(self, app, blocktype, **kwargs):
        if blocktype in config.BLOCK_TYPE_STYLES:
            kwargs.update(config.BLOCK_TYPE_STYLES[blocktype])
        else:
            raise ValueError("invalid block type argument '%s'." % blocktype)

        AppBlock.__init__(self, app, **kwargs)

        self._temporary_text = self.text
        self.blocktype = blocktype
        self.drag_mode = True
        self.moved = False
        self._app = app
        self._highlightrectangle = None
        self.delete_btn = None
        self._context_visible = False
        self._context_box = ContextBlock(app, **kwargs)
        self._context_box.attach(self)
        self._context_box.set_invisible()

        if blocktype in app._plugin_list:
            self._context_box.populate_buttons(app._plugin_list[blocktype])

        self.drag_data = {'x': 0, 'y': 0}
        self.bind('<ButtonPress-1>', self.on_mouse_down)
        self.bind('<ButtonPress-3>', self.show_context)
        self.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.bind('<B1-Motion>', self.on_motion)

    def draw_outline(self):
        """Draws an outline in the block, indicating that it is currently
        selected."""
        self._highlightrectangle = self._app.canvas.create_rectangle(
            self.x - 2, self.y - 2, self.x + config.BLOCK_SIZE + 1,
            self.y + config.BLOCK_SIZE + 1, fill='#000000')

    def remove_outline(self):
        """Removes the outline from the block."""
        if self._highlightrectangle is not None:
            self._app.canvas.delete(self._highlightrectangle)

    def create_delete_btn(self):
        """Creates a rectangle with a X in it in the top-right corner of the
        block to use as a trigger to delete the block."""
        self.delete_btn = self.create_rectangle(
            config.BLOCK_SIZE - 11, 0, config.BLOCK_SIZE - 1, 10,
            fill='#777777', outline='#000000', tags='deleteRect')
        self.tag_bind('deleteRect', '<ButtonPress-1>', self.remove_this)

    def remove_delete_btn(self):
        self.delete(self.delete_btn)
        self.delete_btn = None
        pass

    def remove_this(self, event):
        self.unbind('<ButtonPress-1>')
        self.unbind('<ButtonPress-3>')
        self.unbind('<ButtonRelease-1>')
        self.unbind('<B1-Motion>')
        self._app.remove_block(self)
        self.exit_link()
        pass

    def exit_link(self):
        self.drag_mode = True
        self.remove_outline()
        self.remove_delete_btn()

    def show_context(self, event):
        if not self._context_visible:
            self._context_box.attach(self)
            self._context_box.set_visible()
            self._context_visible = True
        else:
            self._context_box.set_invisible()
            self._context_visible = False
        pass

    def on_mouse_down(self, event):
        if self._app._wheel_visible:
            self._app.remove_block_wheel()
            return

        if not self.drag_mode:
            return

        self.drag_data['x'] = event.x + self.x
        self.drag_data['y'] = event.y + self.y
        self.moved = False

    def on_mouse_up(self, event):
        # box was selected
        if not self.moved:
            # if another box was already selected, link them
            if self._app._link_mode and self.drag_mode is True:
                self._app.link_with(self)
                return

            # if not, switch link
            if self.drag_mode:
                self.drag_mode = False
                self.draw_outline()
                self.create_delete_btn()
                self._app.enter_link_mode(self)

                if self._context_visible:
                    self._context_box.set_invisible()
                    self._context_visible = False
            else:
                self._app.remove_first_link()
                self.exit_link()
            return

        self.drag_data['x'] = 0
        self.drag_data['y'] = 0

    def on_motion(self, event):
        self._app.draw_links()

        if not self.drag_mode:
            return

        tk.Misc.lift(self)

        if self._context_visible:
            self._context_box.set_invisible()
            self._context_visible = False

        dx = event.x + self.x - self.drag_data['x']
        dy = event.y + self.y - self.drag_data['y']

        self.drag_data['x'] = event.x + self.x
        self.drag_data['y'] = event.y + self.y

        self.move(dx=dx, dy=dy)
        self.moved = True
