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

def clamp(value, minV, maxV):
    """Returns value if it is between minV and maxV. Returns minV if value is smaller and maxV if value is larger."""
    return min(max(minV, value), maxV)


def color_string_add(colorString1, colorString2, clampV=True):
    """Adds the color represented by two color strings separately in the R, G and B channels and returns the separated channel values.
    Currently only supports adding two color strings of same size.
    Currently only returns the RGB equivalent, with no alpha."""
    # Checking if string lengths are the same
    if len(colorString1) != len(colorString2):
        raise ValueError, "color strings '%s' and '%s' are not of same length." % (colorString1, colorString2)

    # Removing possible initial '#'
    if colorString1[0] == '#':
        colorString1 = colorString1[1:]
    if colorString2[0] == '#':
        colorString2 = colorString2[1:]

    # Getting ARGB values
    a1, r1, g1, b1 = convert_color_string_to_RGB(colorString1)
    a2, r2, g2, b2 = convert_color_string_to_RGB(colorString2)

    a, r, g, b = a1 + a2, r1 + r2, g1 + g2, b1 + b2

    if clampV:
        # Checking what is the size of the string in order to clamp values accordingly
        if len(colorString1) <= 4:
            a, r, g, b = [clamp(n, 0, 15) for n in (a, r, g, b)]
        elif len(colorString1) >= 6:
            a, r, g, b = [clamp(n, 0, 255) for n in (a, r, g, b)]

    return r, g, b


def color_string_subtract(colorString1, colorString2, clampV=True):
    """Subtracts the color represented by two color strings separately in the R, G and B channels and returns the separated channel values.
    The operation realized is colorString1 - colorString 2.
    Currently only supports adding two color strings of same size.
    Currently only returns the RGB equivalent, with no alpha."""
    # Checking if string lengths are the same
    if len(colorString1) != len(colorString2):
        raise ValueError, "color strings '%s' and '%s' are not of same length." % (colorString1, colorString2)

    # Removing possible initial '#'
    if colorString1[0] == '#':
        colorString1 = colorString1[1:]
    if colorString2[0] == '#':
        colorString2 = colorString2[1:]

    # Getting ARGB values
    a1, r1, g1, b1 = convert_color_string_to_RGB(colorString1)
    a2, r2, g2, b2 = convert_color_string_to_RGB(colorString2)

    a, r, g, b = a1 - a2, r1 - r2, g1 - g2, b1 - b2

    if clampV:
        # Checking what is the size of the string in order to clamp values accordingly
        if len(colorString1) <= 4:
            a, r, g, b = [clamp(n, 0, 15) for n in (a, r, g, b)]
        elif len(colorString1) >= 6:
            a, r, g, b = [clamp(n, 0, 255) for n in (a, r, g, b)]

    return r, g, b


def convert_color_string_to_RGB(colorString):
    """Converts a string representing a color into an ARGB tuple containing the values in integers.
    The string passed can be in any of the following formats: #RGB, #ARGB, #RRGGBB or #AARRGGBB, with our without the starting '#'."""
    # Removing possible initial '#'
    if colorString[0] == '#':
        colorString = colorString[1:]

    # Extracting ARGB strings
    if len(colorString) == 3:
        # no alpha
        a, r, g, b = 'F', colorString[0], colorString[1], colorString[2]
    elif len(colorString) == 4:
        # alpha
        a, r, g, b = colorString[0], colorString[1], colorString[2], colorString[3]
    elif len(colorString) == 6:
        # no alpha
        a, r, g, b = 'FF', colorString[:2], colorString[2:4], colorString[4:]
    elif len(colorString) == 8:
        # alpha
        a, r, g, b = colorString[:2], colorString[2:4], colorString[4:6], colorString[6:]
    else:
        raise ValueError, "input #%s is not in one of the following formats: #RGB, #ARGB, #RRGGBB or #AARRGGBB." % colorString

    a, r, g, b = [int(n, 16) for n in (a, r, g, b)]

    return a, r, g, b


def convert_RGB_to_color_string(r, g, b):
    """Converts RGB values into a 7-character color string, including the starting '#'."""
    colorString = "#%02x%02x%02x" % (r, g, b)
    return colorString
