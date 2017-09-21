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

def return_previous_text(variable):
    """Returns the combobox's or the entry's text to its previous text. Should
    be called when the validation on a combobox or an entry returns False."""
    return lambda: (variable.set(variable.old_text))


def validate_combobox(items, variable):
    """Returns a function for callback that receives a text and validates it
    by checking if the text matches any of the strings passed in the **items**
    argument."""
    return lambda text: _validate_combobox(text, items, variable)


def validate_integer(variable):
    """Returns a function for callback that receives a text and validates it
    by checking if the text only contains numeric digits."""
    return lambda text: _validate_integer(text, variable)


def validate_float(variable):
    """Returns a function for callback that receives a text and validates it
    by checking if the text represents a valid number."""
    return lambda text: _validate_float(text, variable)


def _validate_combobox(text, items, variable):
    """Private."""
    if text in items:
        variable.old_text = text
        return True

    return False


def _validate_integer(text, variable):
    """Private."""
    try:
        int(text)
    except ValueError:
        return False
    else:
        variable.old_text = text
        return True


def _validate_float(text, variable):
    """Private."""
    try:
        float(text)
    except ValueError:
        return False
    else:
        variable.old_text = text
        return True
