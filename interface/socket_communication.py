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

import config
import socket
import threading


callbacks = []
thread_obj = None


class NetworkListener(threading.Thread):
    def __init__(self, callbacks):
        threading.Thread.__init__(self, name="Network Listener")
        self._callbacks = callbacks

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((config.COMMUNICATION_HOST, config.COMMUNICATION_PORT))
        s.listen(0)

        remaining_data = ""

        while True:
            conn, addr = s.accept()

            while True:
                data = conn.recv(1024)

                if not data:
                    break

                msgs = remaining_data + data
                msgs = data.split(config.MESSAGE_SEPARATOR)
                remaining_data = msgs.pop()

                for msg in msgs:
                    if msg is None:
                        continue

                    index = msg.find(' ')
                    msg_id = msg[:index]
                    msg = msg[index + 1:]

                    for item in self._callbacks:
                        if item[0] is None or msg_id in item[0]:
                            item[1](msg_id, msg)

            conn.close()


def register_callback(callback_func, ids=None):
    callbacks.append((ids, callback_func))


def begin_listening():
    thread_obj = NetworkListener(callbacks)
    thread_obj.daemon = True
    thread_obj.start()


def stop_listening():
    pass
