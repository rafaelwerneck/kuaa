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
cyan_err('import common')

import hashfile
import itertools as it
import random
import numpy as np
import sklearn.metrics as skms
import math
import traceback


def printDataset(labels, feats):
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    yellow_err('######################## Dataset informations ########################')
    acs = list(set(labels))
    yellow_err('Total number of samples:       {0}'.format(len(labels)))
    yellow_err('Available classes:             {0}'.format(acs))
    yellow_err('Feature vector size:           {0}'.format(len(feats[0])))
    for clss in acs:
        yellow_err('Number of samples of the class {0}: {1}'.format(clss, labels.count(clss)))
    yellow_err('######################################################################')


def loadFeatures(file_name):
    """
    Input:
    file_name :: String
    file_name: The name of the file containing the feature vectors.
    Each element must be separated by white space.
    The first element is must be an integer n, such that n > 0,
    indicating the label of the features.

    Output:
    labels :: np.array([Int])
    labels: Array of labels.
    feats :: np.array([[Float]])
    feats: Each internal vector is a feature vector.
    """
    assert isinstance(file_name, str)

    import numpy as np

    with open(file_name) as fp:
        data = fp.readlines()

        data = np.array(map(lambda x: map(float, x.split()), data))

        labels = data[:, 0].astype('int')
        feats = data[:, 1:]

    assert isinstance(labels, np.ndarray)
    assert isinstance(feats, np.ndarray)
    assert len(set(map(len, feats))) == 1

    return labels, feats


def splitTrainTestOS(labels, feats, acs):
    """
    Split 'labels' and 'feats' in train and test for
        open set (OS) evaluation.
    """
    assert isinstance(labels, np.ndarray)
    assert isinstance(feats, np.ndarray)
    assert isinstance(acs, list)
    assert len(labels) == len(feats)
    assert len(acs) > 0
    assert all([x in labels
                for x
                in acs])

    laf = zip(labels, feats)  # labels and feats
    train = filter(lambda (l, f): l in acs, laf)

    grouped = map(lambda x: list(x[1]),
                  it.groupby(train, lambda (l, f): l))  # grouped based on labels

    try:
        hf = current_simulations_settings['hf']
        sf = hf.getStorageFile([labels, feats, acs])
        try:
            seeds_shuffle = sf.loadVars('seeds_shuffle')
            yellow_err('splitTrainTestOS(): try')
        except:
            yellow_err('splitTrainTestOS(): except')
            seeds_shuffle = [random.random()
                             for _ in range(len(grouped))]
            sf.saveVars(seeds_shuffle=seeds_shuffle)
        finally:
            del hf
            del sf
    except NameError:
        seeds_shuffle = [random.random()
                         for _ in range(len(grouped))]

    map(lambda (group, seed): random.shuffle(group, lambda: seed),
        zip(grouped, seeds_shuffle))  # shuffle feats of the same label among them
    grouped = map(lambda group: (group[:len(group) / 2],
                            group[len(group) / 2:]),
                  grouped)  # split each group of some label between train and test

    train, test = reduce(lambda (tr1, te1), (tr2, te2): (tr1 + tr2, te1 + te2),
                         grouped)
    test += map(lambda (l, f): (None, f),
                filter(lambda (l, f): l not in acs,
                       laf))

    ltr, ftr = map(np.array, zip(*train))
    lte, fte = map(np.array, zip(*test))

    assert len(ltr) + len(lte) == len(labels)
    assert len(ftr) + len(fte) == len(feats)

    return ltr, ftr, lte, fte


def shuffleData(labels, feats):
    assert len(labels) == len(feats)

    indexes = range(len(labels))
    random.shuffle(indexes)

    def simpleShuffle(lst):
        return map(lambda i: lst[i], indexes)

    labels = simpleShuffle(labels)
    feats = simpleShuffle(feats)

    return labels, feats


def getFold(labels, feats, k, n, fold_type):
    assert len(labels) == len(feats)

    if fold_type == 'train':
        pass
    elif fold_type == 'test':
        pass


def scaleData(ftr, fte=None):
    """
    np.array(feats_train) -> np.array(feats_test) -> (np.array(feats_train), np.array(feats_test))

    The resulting features will be in [0, 1].
    """
    assert isinstance(ftr, np.ndarray)
    if fte is not None:
        assert isinstance(fte, np.ndarray)

    max_column = ftr.max(axis=0)
    min_column = ftr.min(axis=0)
    dif_column = (max_column - min_column).astype('float')
    dif_column[dif_column == 0] = 1

    ftr2 = (ftr - min_column) / dif_column
    if fte is not None:
        fte2 = (fte - min_column) / dif_column

    assert isinstance(ftr2, np.ndarray)
    if fte is not None:
        assert isinstance(fte2, np.ndarray)

    if fte is not None:
        return ftr2, fte2
    else:
        return ftr2


def limitTraining(lv, fv, acs):
    """
    This function is useful for training considering the open set scenario.
    The labels from 'lv' and its corresponding feature from 'fv' are maintained only if
        that label is in 'acs'.

    Input:
        lv :: [Float]
        lv: Label vector.
        fv :: [[Float]]
        fv: Vector of feature vectors.
        acs :: [Float]
        acs: Available classes. The classes to train the classifier with.

    Output:
        (lv, fv) :: ([Float], [[Float]])
    """
    assert len(lv) == len(fv)

    lv, fv = map(list,
                 zip(*[(l, f)
                       for l, f
                       in zip(lv, fv)
                       if l in acs]))

    assert isinstance(lv, list)
    assert isinstance(fv, list)

    return lv, fv


#===============================================================================
# Functions related to the acquisition of the results, based on the predicted
# labels.
#===============================================================================

def predictionResults(lte, lpr, unknown_label=None, have_unknown=False, pls_binary_lpr=None):
    """
    Input:
        lte :: [Float]
        lpr :: [Float]

    Output:
    """
    assert len(lte) == len(lpr)

    if unknown_label is not None:
        assert unknown_label not in lte
        assert unknown_label not in lpr
    else:
        unknown_label = min(list(set([x for x in lte if x is not None])
                                 .union(set([x for x in lpr if x is not None])))) - 999

    """
    If some label appears either in the testing set or in
        prediction, it is considered in results acquisition.

    Note that if the "unknown label" appears in 'lte', then
        'list(set(lte).union(set(lpr))) == list(set(lte))'.
    """
    lte = [x
           if x is not None
           else unknown_label
           for x in lte]
    lpr = [x
           if x is not None
           else unknown_label
           for x in lpr]

    labels = list(set(lte).union(set(lpr)))
    try:
        labels.remove(unknown_label)
        labels.append(unknown_label)  # to be the last element on the list
        have_unknown = True
    except:
        have_unknown = max(False, have_unknown)

    """
    bbfm = binary-based f-measure
    """
    if pls_binary_lpr is not None:
        pls, binary_lpr = pls_binary_lpr
        tls = [None if l == unknown_label else l
               for l in lte]
        tls, pls = np.meshgrid(tls, pls)
        def f(x): return x.reshape(1, x.size).tolist()[0]
        tls, pls, binary_lpr = f(tls), f(pls), f(np.array(binary_lpr))
        def g((tl, pl, blpr)):
            """
            tl: true label or test label
            pl: positive label
            blpr: binary label of the prediction
            """
            assert blpr is None or blpr == pl

            if blpr == pl and blpr == tl:
                return 'TP'
            elif blpr == pl and blpr != tl:
                return 'FP'
            elif blpr is None and tl is None:
                return 'TN'
            elif blpr is None and tl is not None:
                return 'FN'
            else:
                raise ValueError('Unrecognized configuration.')
        aux = map(g, zip(tls, pls, binary_lpr))
        TP = float(aux.count('TP'))
        FP = float(aux.count('FP'))
        TN = float(aux.count('TN'))
        FN = float(aux.count('FN'))
        precision = TP / (TP + FP) if TP != 0.0 else 0.0
        recall = TP / (TP + FN) if TP != 0.0 else 0.0
        bbfm = (0.0
                if precision == 0.0 and recall == 0.0
                else (2.0 * precision * recall) / (precision + recall))
    else:
        bbfm = None

    if not have_unknown:
        red_err('predictionResults(): have_unknown = {0}'.format(have_unknown))

    cm = skms.confusion_matrix(lte, lpr, labels).astype(float)  # confusion matrix

    if not have_unknown:
        raise Exception('For now, I do not expect to acquire results without unknown.')
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]
        precision = TP / (TP + FP) if TP != 0.0 else 0.0  # Porcentagem que realmente eh positivo, dentro dos que falei que eram
        recall = TP / (TP + FN) if TP != 0.0 else 0.0  # Porcentagem do que eu falei que eh positivo, dentro de todos os positivos

        sensitivity = TP / (TP + FN) if TP != 0.0 else 0.0  # Porcentagem do que eu falei que eh positivo, dentro de todos os positivos
        specificity = TN / (TN + FP) if TN != 0.0 else 0.0  # Porcentagem do que eu falei que eh negativo dentro de todos os negativos
        naccuracy = (sensitivity + specificity) / 2.0
        fmeasure = 0.0 if precision == 0.0 and recall == 0.0 else (2.0 * precision * recall) / (precision + recall)

        return naccuracy, sensitivity, specificity, fmeasure

    else:
        known_data = [(lp, lt) for lp, lt in zip(lpr, lte) if lt != unknown_label]
        unknown_data = [(lp, lt) for lp, lt in zip(lpr, lte) if lt == unknown_label]

        known_accuracy = sum([lp == lt for lp, lt in known_data]) / float(len(known_data))
        unknown_accuracy = sum([lp == lt for lp, lt in unknown_data]) / float(len(unknown_data)) if len(unknown_data) > 0 else 1.0
        error = sum([lp != lt for lp, lt in known_data]) / float(len(known_data))
        misclassification = sum([lp != lt and lp != unknown_label for lp, lt in known_data]) / float(len(known_data))
        false_unknown = sum([lp != lt and lp == unknown_label for lp, lt in known_data]) / float(len(known_data))
        true_unknown = unknown_accuracy

        accuracy_per_class = [np.mean([lp == lt
                                       for lp, lt
                                       in zip(lpr, lte)
                                       if lt == l])
                              for l
                              in labels]
        yellow_err('predictionResults(): accuracy_per_class: {0}'.format(zip(labels, accuracy_per_class)))
        global_na = float(np.mean(accuracy_per_class))
        yellow_err('predictionResults(): labels, global_na, na = {0}, {1}, {2}'.format(labels, global_na, (known_accuracy + unknown_accuracy) / 2))

        def precisionMacro(indexes):
            def f(i):
                TP = cm[i, i]
                FP = sum(cm[:, i]) - TP
                val = TP / (TP + FP) if TP != 0.0 else 0.0
                return val
            precision = np.mean(map(f, indexes))
            return precision

        def recallMacro(indexes):
            def f(i):
                TP = cm[i, i]
                FN = sum(cm[i, :]) - TP
                val = TP / (TP + FN) if TP != 0.0 else 0.0
                return val
            recall = np.mean(map(f, indexes))
            return recall

        def precisionMicro(indexes):
            def f1(i):
                TP = cm[i, i]
                return TP

            def f2(i):
                TP = cm[i, i]
                FP = sum(cm[:, i]) - TP
                return TP + FP

            num, den = sum(map(f1, indexes)), sum(map(f2, indexes))
            precision = num / den if den != 0 else 0.0
            return precision

        def recallMicro(indexes):
            def f1(i):
                TP = cm[i, i]
                return TP

            def f2(i):
                TP = cm[i, i]
                FN = sum(cm[i, :]) - TP
                return TP + FN

            num, den = sum(map(f1, indexes)), sum(map(f2, indexes))
            recall = num / den if den != 0 else 0.0
            return recall

        # Calculate the MAFM and MIFM.

        indexes = range(len(labels) - 1)
        precision_macro = precisionMacro(indexes)
        recall_macro = recallMacro(indexes)
        precision_micro = precisionMicro(indexes)
        recall_micro = recallMicro(indexes)

        fmeasure_macro = ((2.0 * precision_macro * recall_macro) / (precision_macro + recall_macro)
                          if precision_macro != 0.0 or recall_macro != 0.0
                          else 0.0)
        fmeasure_micro = ((2.0 * precision_micro * recall_micro) / (precision_micro + recall_micro)
                          if precision_micro != 0.0 or recall_micro != 0.0
                          else 0.0)

        # Print

        yellow_err(reduce(lambda a, b: a + '.' + b, map(lambda x: str(x[2]) + '()', traceback.extract_stack())))
        yellow_err('predictionResults(): pma, rma, pmi, rmi, (fma, fmi) = {0}, {1}, {2}, {3}, ({4}, {5})'.format(precision_macro, recall_macro, precision_micro, recall_micro, fmeasure_macro, fmeasure_micro))

        # Calculate the NAKS

        naks_labels = labels[:]
        try:
            naks_labels.remove(unknown_label)
        except:
            pass
        naks = [np.mean([lp == lt
                         for lp, lt
                         in zip(lpr, lte)
                         if lt == l])
                for l
                in naks_labels]
        naks = float(np.mean(naks))

        # Calculate the CSMAFM and CSMIFM

        indexes = range(len(labels))
        precision_macro = precisionMacro(indexes)
        recall_macro = recallMacro(indexes)
        precision_micro = precisionMicro(indexes)
        recall_micro = recallMicro(indexes)

        csmafm = ((2.0 * precision_macro * recall_macro) / (precision_macro + recall_macro)
                  if precision_macro != 0.0 or recall_macro != 0.0
                  else 0.0)
        csmifm = ((2.0 * precision_micro * recall_micro) / (precision_micro + recall_micro)
                  if precision_micro != 0.0 or recall_micro != 0.0
                  else 0.0)

        ret = {
            'ERR': error,             # ERRor
            'MIS': misclassification, # MISclassification
            'FU': false_unknown,      # False Unknown
            'AKS': known_accuracy,    # Accuracy of Known Samples
            'AUS': unknown_accuracy,  # Accuracy of Unknown Samples
            'MAFM': fmeasure_macro,   # MAcro-averaging F-Measure
            'MIFM': fmeasure_micro,   # MIcro-averaging F-Measure
            'NA': (known_accuracy + unknown_accuracy) / 2, # Normalized Accuracy
            'GNA': global_na,   # Global Normalized Accuracy
            'NAKS': naks,       # Normalized Accuracy of Known Samples
            'BBFM': bbfm,       # Binary-Based F-Measure
            'CSMAFM': csmafm,     # Closed-Set MAcro-averaging F-Measure
            'CSMIFM': csmifm,     # Closed-Set MIcro-averaging F-Measure
            }

        if False:
            return known_accuracy, unknown_accuracy, np.mean(fmeasure)
        else:
            return ret

def mymeshgrid(*lst):
    """
    "Nao me pergunte por que
    Quem-Como-Onde-Qual-Quando-O que?"
    """

    if np.__version__ >= '1.7':
        if len(lst) > 1:
            ret = map(list,
                      zip(*map(lambda x: x.flatten(),
                               np.meshgrid(*lst))))
        else:
            ret = lst[:]

    else:
        class caux:
            def __init__(self, values):
                self.values = values
            def __repr__(self):
                return 'caux: ' + repr(self.values)

        def f(caux1, caux2):
            return map(lambda (a, b): caux(a.values + b.values),
                       zip(*map(lambda x: x.flatten(),
                                np.meshgrid(caux1, caux2))))

        def g(lst):
            return map(lambda x: caux([x]), lst)

        def extract(lst):
            return map(lambda x: x.values, lst)

        ret = extract(reduce(f, map(g, lst)))

    return ret

def gridSearch(iterations, max_or_min, of, *params): # PEDRORMJUNIOR: 20130806: add max_or_min
    """
    Input:
    iterations = number of iterations
    of = objective function.  A function that receives 'n' parameters and return an accuracy measure.
    *params = lists of parameters.  The first list constains values to be grid searched as the first parameters of the 'of'.  The second list contains values to be grid searched as the second parameters of the 'of'.  And so on...

    Output:
    ret = a list of parameters of size 'n'.  The best combination of parameters according to the 'of'.
    """
    yellow_err('gridSearch(): iterations = {0}'.format(iterations))

    assert max_or_min in ['min', 'max']

    lenranges = map(len, params)
    assert all(map(lambda x: x > 1, lenranges))
    positions = map(lambda i: int(np.prod(lenranges[:i])),
                    range(len(lenranges)))

    meshparams = mymeshgrid(*params)

    results = []
    for i in range(len(meshparams)):
        res = (of(*meshparams[i]),
               (-i if max_or_min == 'min' else i),
               i)
        results.append(res)
    _, _, best_params_idx = max(results)
    yellow_err('gridSearch(): best_params = {0}'.format(meshparams[best_params_idx]))

    if iterations == 0:
        ret = meshparams[best_params_idx]
    else:
        shifts = map(lambda i: abs(meshparams[0][i] - meshparams[positions[i]][i]) / 2.0,
                     range(len(lenranges)))

        newparams = map(lambda i: np.linspace(meshparams[best_params_idx][i] - shifts[i],
                                         meshparams[best_params_idx][i] + shifts[i],
                                         num=lenranges[i]).tolist(),
                        range(len(lenranges)))

        ret = gridSearch(iterations - 1, max_or_min, of, *newparams)

    return ret


def gridSearchExponential(max_or_min, of, *params): # PEDRORMJUNIOR: 20130806: add max_or_min
    """
    Input:
    of = objective function.  A function that receives 'n' parameters and return an accuracy measure.
    *params = lists of parameters.  The first list constains values to be grid searched as the first parameters of the 'of'.  The second list contains values to be grid searched as the second parameters of the 'of'.  And so on...

    Output:
    ret = a list of parameters of size 'n'.  The best combination of parameters according to the 'of'.
    """
    assert max_or_min in ['min', 'max']

    lenranges = map(len, params)
    assert all(map(lambda x: x > 1, lenranges))
    positions = map(lambda i: int(np.prod(lenranges[:i])),
                    range(len(lenranges)))

    meshparams = mymeshgrid(*params)

    results = []
    for i in range(len(meshparams)):
        try:                    # Maybe, the params are not valid.
            res = (of(*meshparams[i]),
                   (-i if max_or_min == 'min' else i),
                   i)
            results.append(res)
        except Exception as expt:
            red_err(traceback.format_exc()[:-1])
            red_err('gridSearch: except: i, params = {0} {1}'.format(i, meshparams[i]))

    _, _, best_params_idx = max(results)
    yellow_err('gridSearch(): best_params = {0}'.format(meshparams[best_params_idx]))

    ret = meshparams[best_params_idx]

    return ret


def putUnknownLabel(lte, acs):
    """
    Unknown classes receive the label 0.
    """
    assert isinstance(lte, np.ndarray)

    indexes = np.zeros(len(lte))
    for i in acs:
        indexes = np.logical_or(indexes, lte == i)
        lte = lte.copy()        # Must be done.
        lte[np.logical_not(indexes)] = 0

        return lte
