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

from termcolor import *
import sys
cyan_err('import OPF')

import hashfile
import numpy as np
from scipy.spatial import distance
import itertools as it
import mlpy
import random
import common
import pylab as pl
from heapq import heappush, heappop, heapify
import copy
from Classifier import Classifier

class OPF(Classifier):
    def simple__init__(self):
#         assert False            # I don't want to use it.
        self._gridsearch = False
        
        #Werneck
        #-------
        assert self._parameters.has_key('distance')
        self._distance = self._parameters['distance']
        assert isinstance(self._distance, str)
        #-------

    def simple__repr__(self):
        return 'OPF'

    def simplefit(self):
        Classifier.simplefit(self)

        #Werneck
        #-------
        try:
            graph = distance.squareform(distance.pdist(self._vftr, self._distance))
        except:
            try:
                descriptor_path = os.path.join(os.path.dirname(__file__), '..',
                        '..', 'descriptors', self._distance)
                sys.path.append(descriptor_path)
                software = __import__('plugin_' + self._distance)
                print software
                graph = distance.squareform(distance.pdist(self._vftr, software.distance))
            except:
                graph = distance.squareform(distance.pdist(self._vftr))
        #-------

        mst = self.__prim(graph)  # mst = minimum spanning tree
        prototypes = self.__findPrototypes(mst, self._vltr)
        prototypes = np.array(prototypes)

        vltr = np.array(self._vltr)
        prototypes_labels = vltr[prototypes]

        C, P, L = self.__dijkstra(mst, prototypes, prototypes_labels)

        # Storing data on the self.
        self.C, self.P, self.L = C, P, L
        self.vftr, self.vltr = self._vftr, self._vltr

        assert len(self.vftr) == len(self.L)
        assert len(self.L) == len(self.C)

    def _preAssertPredict(self, vte):
        assert self._trained
        assert len(vte) == len(self.ftr[0])

    def predict(self, vte):
        """
        """
        W = distance.cdist([vte], self.ftr)[0]
        lcw = zip(self.L, self.C, W)  # label, cost, weight
        nf = min(lcw, key=lambda (l, c, w): max(c, w))  # nearest feature

        ret = nf[0]

        return ret

    def simpleclassify(self, auxiliar=None):
        """
        """
        Classifier.simpleclassify(self, auxiliar=None)

        #Werneck
        #-------
        try:
            lpr = map(lambda (i, _): self.L[i],
                      map(lambda enumdists: min(enumdists, key=lambda (i, dist): max(dist, self.C[i])),
                          map(enumerate, distance.cdist(self._vfte, self.vftr, self._distance))))
        except:
            try:
                descriptor_path = os.path.join(os.path.dirname(__file__), '..',
                        '..', 'descriptors', self._distance)
                sys.path.append(descriptor_path)
                software = __import__('plugin_' + self._distance)
                print software
                lpr = map(lambda (i, _): self.L[i],
                          map(lambda enumdists: min(enumdists, key=lambda (i, dist): max(dist, self.C[i])),
                              map(enumerate, distance.cdist(self._vfte, self.vftr, software.distance))))
            except:
                lpr = map(lambda (i, _): self.L[i],
                          map(lambda enumdists: min(enumdists, key=lambda (i, dist): max(dist, self.C[i])),
                              map(enumerate, distance.cdist(self._vfte, self.vftr))))
        #-------

#         if lte is not None:     # For plotting purpose
        if False:
            lst_enum_costs = map(lambda enumdists:
                                     map(lambda (j, dist): (j, max(dist, self.C[j])),
                                         enumerate(enumdists)),
                                 distance.cdist(self._vfte, self.vftr))
            nfs = map(lambda (j, cost): (self.L[j], cost),
                      map(lambda enumcosts: min(enumcosts, key=lambda (j, cost): cost),
                          lst_enum_costs))
            labels = np.array(map(lambda (l, c): l, nfs), ndmin=2).transpose()
            costs = np.array(map(lambda (l, c): c, nfs), ndmin=2).transpose()

            while labels.shape[1] < len(self._acs):
                nfols = map(lambda (j, cost): (self.L[j], cost),
                            map(lambda (k, enumcosts): min(enumcosts, key=lambda (j, cost): cost if self.L[j] not in labels[k, :] else float('inf')),
                                enumerate(lst_enum_costs)))
                labels = np.append(labels, np.array(map(lambda (l, c): l, nfols), ndmin=2).transpose(), axis=1)
                costs = np.append(costs, np.array(map(lambda (l, c): c, nfols), ndmin=2).transpose(), axis=1)

            colors = ['blue', 'red', 'green', 'black']
             # markers = ['D', 'o', 's', '.', 'v', '>', '+', ',', '^', '<', 's']

            fig = pl.figure()
            x = range(len(self._acs))

            positive_costs, negative_costs, unknown_costs = [], [], []
            counterb, counterr, counterg = 0, 0, 0
            mymax = 10
            for k in range(len(costs)):
                if labels[k, 0] == self._vlte[k]:
                    positive_costs.append(costs[k, 0])
                    if counterb < mymax:
                        pl.plot(x, costs[k], colors[0])
                        counterb += 1
                elif self._vlte[k] != 0:
                    negative_costs.append(costs[k, 0])
                    if counterr < mymax:
                        pl.plot(x, costs[k], colors[1])
                        counterr += 1
                else:
                    unknown_costs.append(costs[k, 0])
                    if counterg < mymax:
                        pl.plot(x, costs[k], colors[2])
                        counterg += 1

            fig.show()
            raw_input('ENTER;')

            fig = pl.figure()
            bins = tuple([y / 2.0 for y in range(0, 41)])
            pl.hist(positive_costs, bins=bins, normed=True, color='b', histtype='step', label='Positive')
            pl.hist(negative_costs, bins=bins, normed=True, color='r', histtype='step', label='Negative')
            pl.hist(unknown_costs, bins=bins, normed=True, color='g', histtype='step', label='Unknown')
            pl.legend(loc='best')
            fig.show()
            raw_input('ENTER;')

        return lpr

    def __dijkstra(self, A, seeds, seeds_labels):
        """
        Parameters:
            A = the adjacency matrix of a graph
            seeds = the indices of the nodes to be used as seeds of the SPF-Max
            seeds_labels = labels of the seeds.
        Note:
            This function corresponds to the multi-source-max Dijkstra algorithm.
        Return:
            c1 = costs to the corresponding parent root
            par = parents nodes
            lab = labels of the nodes
        """
        h = 0
        n = A.shape[0]
        done = np.zeros(n).astype('bool')
        c1  = np.ones(n) * float("inf")
        par = np.ones(n) * -1
        lab = np.ones(n) * float("inf")
        heap = []
        for i, v0 in enumerate(seeds):
            c1[v0] = h
            par[v0] = v0
            lab[v0] = seeds_labels[i]
            heappush(heap, (h, v0))

        while len(heap) > 0:
            (_, p) = heappop(heap)
            done[p] = True
            for q in xrange(n):
                if p == q or done[q]:
                    continue
                c = max(c1[p], A[p, q])  # fcost
                if c < c1[q]:
                    if c1[q] < float("inf"):
                        try:
                            heap.remove((c1[q], q))
                            heapify(heap)
                        except:
                            pass
                    c1[q]  = c
                    par[q] = p
                    lab[q] = lab[p]
                    heappush(heap, (c1[q], q))

        return c1, par, lab

#   @staticmethod
    def __prim(self, A):
        """
        Parameters:
        A = Weighted adjacency matrix. Non connected nodes have weigh 'Inf'.
        Note:

        Return:
        B =  weighted adjacency matrix corresponding to the MST of A.
        """
        root = 0
        # prevents zero-weighted MSTs
        A = A.astype('double')

        n = A.shape[0]
        if n == 0: return []

        d = np.zeros(n)         # type(d) = numpy.ndarray
        pred = np.zeros(n, int)  # type(pred) = numpy.ndarray
        B = np.ones(A.shape) * float("inf")  # type(B) = numpy.ndarray

        b0 = root         # first vertex as root. type(b0) = int
        Q = range(0, n)   # all vertices. type(Q) = list
        Q.remove(root)    # except the root.
        d[Q] = A[Q, b0]   # distances from root.
        pred[Q] = b0      # predecessor as root.

        while Q != []:
            s = np.argmin(d[Q])  # function `argmin' in module `numpy.core.fromnumeric'.
            closest_i = Q[s]  # find vertex in Q with smallest weight from b0
            Q.remove(closest_i)
            b0 = closest_i
            b1 = pred[closest_i]
            B[b0, b1] = B[b1, b0] = A[b0, b1]  # insert this edge in MST weight matrix
            if Q == []: break
            QA = np.array(Q)    # remaining vertexes.
            iii = np.where(A[QA, closest_i] < d[QA])
            d[QA[iii]] = A[QA[iii], closest_i]     # update weights to pred
            pred[QA[iii]] = closest_i             # update pred

        return B

#   @staticmethod
    def _binDecision(self, LM, CM, check):
        """
        Parameters:
        LM = Labels matrix, each line corresponds  to one of OPF's binary results labels
        CM = Costs matrix, each line corresponds  to one of OPF's binary results costs
        check = {1,2,3,4} -> Choice of method for solving the problem of multiple TP
        Note:
        Surprisingly  the best check for the SCA problem was the number 4, which chooses between the multiple positives,
        the label of highest cost.
        Return:
        labels = labels vector
        """
        _, c = LM.shape
        labels = np.zeros(c)
        CM = CM.transpose()
        # Does it make sense to normalize the costs of all samples for one classifiers?
#         max_column = CM.max(axis=0)
#         min_column = CM.min(axis=0)
# #         print max_column
# #         print min_column
#         dif_column = max_column - min_column
# #         print dif_column
#         CM = (CM - min_column) / dif_column
        CM = CM.transpose()

        if check == 1:
            labels = LM[CM.argmin(axis=0), range(c)]

        # For one sample: among the several classifiers that
        # classified the sample as known (one of the available
        # classes), choose the label in which the cost is the
        # smallest.  If all classifiers classify as unknown, then
        # classify as unknown.
        elif check == 2:
            for i in range(c):
                indexes = np.where(LM[:, i] != self._unknown_label)[0]
                if len(indexes) == 0:
                    labels[i] = self._unknown_label
                elif len(indexes) == 1:
                    labels[i] = LM[indexes, i]
                else:
                    temp_LM = LM[indexes, i]
                    temp_CM = CM[indexes, i]
                    index = temp_CM.argmin()
                    labels[i] = temp_LM[index]

        elif check == 3:
            for i in range(c):
                indexes = np.where(LM[:, i] != 0)[0]
                if len(indexes) > 1:
                    labels[i] = 0
                elif len(indexes) == 0:
                    labels[i] = 0
                else:
                    labels[i] = LM[indexes, i]

        elif check == 4:
            for i in range(c):
                indexes = np.where(LM[:, i] != 0)[0]
                if len(indexes) == 0:
                    labels[i] = 0
                elif len(indexes) == 1:
                    labels[i] = LM[indexes, i]
                else:
                    temp_LM = LM[indexes, i]
                    temp_CM = CM[indexes, i]
                    index = temp_CM.argmax()
                    labels[i] = temp_LM[index]
        else:
            raise ValueError

        return labels.astype(int)

#   @staticmethod
    def __findPrototypes(self, MST, labels):
        """
        Parameters:
        MST = MST adjacency matrix
        labels = labels of the nodes
        Note:

        Return:
        seeds = List with OPF prototypes
        """
        n = MST.shape[0]
        if n != len(labels): return []

        seeds = []

        for i in range(n):
            w = MST[i, :]
            wix = np.where(w != float("inf"))[0]
            l = map(lambda x: labels[x], wix)
            if set(l + [labels[i]]).__len__() > 1:
                seeds.append(i)

        return seeds
