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

# : Constant to identify collection blocks
COLLECTION = "collection"

# : Constant to identify train/test blocks
TRAINTEST = "train_test_method"

# : Constant to identify feature extraction blocks
FEATURE = "descriptor"

# : Constant to identify normalizer blocks
NORMALIZER = "normalizer"

# : Constant to identify classifier blocks
CLASSIFIER = "classifier"

# : Constant to identify fusion blocks
FUSION = "fusion_method"

# : Constant to identify evaluation measure blocks
EVALUATION = "evaluation_measure"

APPLICATION_BLOCKS = [COLLECTION, TRAINTEST, FEATURE, NORMALIZER, CLASSIFIER, FUSION, EVALUATION]

WHEEL_RADIUS = 150
BLOCK_SIZE = 80

COMMUNICATION_HOST = '127.0.0.1'
COMMUNICATION_PORT = 49981

# MESSAGE IDS
MESSAGE_MODULE_FOLDER = '0'
MESSAGE_MODULE_START = '1'
MESSAGE_MODULE_PROGRESS = '2'
MESSAGE_EXPERIMENT_FINISH = '3'
MESSAGE_ITERATION_FINISH = '4'
MESSAGE_SEPARATOR = '///'

STAGE_BACKGROUND = '#AAAAAA'

LINK_ATTRIBUTES = {'width': 3,
            'tags': ('linkline',),
            'capstyle': tk.PROJECTING,
            'arrow': tk.LAST,
            'arrowshape': (12, 15, 9)}

# : Contains all style options for each block type
BLOCK_TYPE_STYLES = {
    COLLECTION: {
        'color1': '#FE9494',
        'color2': '#CD0000',
        'text': 'COLLECTION'
    },
    FEATURE: {
        'color1': '#B1FFE1',
        'color2': '#009494',
        'text': 'FEATURE\nEXTRACTION'
    },
    NORMALIZER: {
        'color1': '#94D7FE',
        'color2': '#0080CD',
        'text': 'NORMALIZER'
    },
    CLASSIFIER: {
        'color1': '#FEE494',
        'color2': '#B18400',
        'text': 'CLASSIFIER'
    },
    FUSION: {
        'color1': '#D8B1FF',
        'color2': '#6600CD',
        'text': 'FUSION'
    },
    EVALUATION: {
        'color1': '#94FED7',
        'color2': '#00B16E',
        'text': 'EVALUATION\nMEASURE'
    },
    TRAINTEST: {
        'color1': '#FFCCFF',
        'color2': '#B388B3',
        'text': 'TRAIN/TEST'
    }
}
