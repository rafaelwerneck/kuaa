
from termcolor import *
import sys
cyan_err('import SVM')

import svmutil
import svmutil1vs
import numpy as np
import common
import hashfile
import mlpy
import random
from Classifier import Classifier


def svm_training_function(vltr, vftr, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    all_parameters = ' -q -s 0 -t 2 ' + parameters
    yellow_err('svm_training_function(): parameters = {0}'.format(all_parameters))
    return svmutil.svm_train(vltr, vftr, all_parameters)


def svm_dbc_training_function(vltr, vftr, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    all_parameters = ' -q -s 0 -t 2 ' + parameters
    yellow_err('svm_dbc_training_function(): parameters = {0}'.format(all_parameters))
    return svmutil.svm_train_dbc(vltr, vftr, all_parameters)


def svm_1vs_training_function(vltr, vftr, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    all_parameters = ' -q -s 7 ' + parameters
    yellow_err('svm_1vs_training_function(): parameters = {0}'.format(all_parameters))
    ret = svmutil1vs.svm_train(vltr, vftr, all_parameters)
    return ret


def svm_prediction_function(vltr, vftr, model, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    all_parameters = ' -q ' + parameters
    yellow_err('svm_prediction_function(): parameters = {0}'.format(all_parameters))
    return svmutil.svm_predict(vltr, vftr, model, all_parameters)


def svm_1vs_prediction_function(vltr, vftr, model, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    all_parameters = ' -q ' + parameters
    yellow_err('svm_1vs_prediction_function(): parameters = {0}'.format(all_parameters))
    ret = svmutil1vs.svm_predict(vltr, vftr, model, all_parameters)
    return ret


# BERNARDO BEGIN (18/08/2013)
def svm_sh_training_function(vltr, vftr, parameters, numClassifiers = 0):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    return svmutil.svm_train_sh(vltr, vftr, ' -q ' + parameters, numClassifiers)


def svm_sh_prediction_function(vltr, vftr, model, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    return svmutil.svm_predict_sh(vltr, vftr, model, ' -q ' + parameters)
# BERNARDO END (18/08/2013)


"""
I must define this function here.  This function cannot be define
inside 'OCSVM' class.  Otherwise, it will raise an Exception when
using the 'multiprocessing' library.
"""
def one_class_svm_training_function(vltr, vftr, parameters):
    assert isinstance(vltr, list)
    assert isinstance(vftr, list)
    assert isinstance(parameters, str)
    return svmutil.svm_train(vltr, vftr, ' -s 2 ' + parameters)


class SVM(Classifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert self._parameters.has_key('training_function')
        assert self._parameters.has_key('prediction_function')

        self._training_function = self._parameters['training_function']
        self._prediction_function = self._parameters['prediction_function']
#         if self._parameters.has_key('C'):
#             self._C = self._parameters['C']
#         if self._parameters.has_key('gamma'):
#             self._gamma = self._parameters['gamma']

        self._gridsearch = False

    def simple__repr__(self):
        return 'SVM interface'

    @staticmethod
    def _featureVectorFromListToDict(lst):
        ret = {idx + 1: value
               for idx, value
               in enumerate(lst)}

        return ret

    def simplefit(self):
        Classifier.simplefit(self)

        vftr = map(SVM._featureVectorFromListToDict, self._vftr)

        vltr = [label
                if label is not None
                else self._unknown_label
                for label in self._vltr]
        str_parameters = ((' -c {0} '.format(self._parameters['C'])
                           if self._parameters.has_key('C')
                           else '') +
                          (' -g {0} '.format(self._parameters['gamma'])
                           if self._parameters.has_key('gamma')
                           else '') +
                          (' -t {0} '.format(self._parameters['kernel_type'])
                           if self._parameters.has_key('kernel_type')
                           else '') +
                          (' -G {0} {1} '.format(self._parameters['near_pressure'], self._parameters['far_pressure'])
                           if self._parameters.has_key('near_pressure') and self._parameters.has_key('far_pressure')
                           else '')
                          )
        self._model = self._training_function(vltr, vftr, ' -q ' + str_parameters)

    def simpleclassify(self, auxiliar=None):
        Classifier.simpleclassify(self, auxiliar)

        vfte = map(SVM._featureVectorFromListToDict, self._vfte)

        vlte = [label
                if label is not None
                else self._unknown_label
                for label in self._vlte]
        vlpr, apr, vpr = self._prediction_function(vlte, vfte, self._model, ' -q ')
        vlpr = [label
                if label != self._unknown_label
                else None
                for label in vlpr]

        if auxiliar is not None:
            for x in vpr:
                auxiliar.append(x)

        assert len(vlpr) == len(vfte)

        del vfte
        del vlte

        return vlpr


class BSVM(SVM):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('training_function')
        assert not self._parameters.has_key('prediction_function')

        self._parameters['training_function'] = svm_training_function
        self._parameters['prediction_function'] = svm_prediction_function

        SVM.simple__init__(self)

        if self._parameters.has_key('C') and self._parameters.has_key('gamma'):
            self._gridsearch = False
        else:
#             assert not self._parameters.has_key('C')
#             assert not self._parameters.has_key('gamma')
            self._gridsearch = True

    def simple__repr__(self):
        return 'BSVM'

    def simple__gridsearch__(self):
        Classifier.simple__gridsearch__(self)

        #Werneck
        #hf = current_simulations_settings['hf']
        hash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage"))
        if not os.path.exists(hash_path):
            os.mkdir(hash_path)
        hf = hashfile.HashFile(hash_path + os.sep, 'HashFile.npz')
        #-------
        sf = hf.getStorageFile([self._acs, self._vftr, self._vltr])

        try:
            itr, iva = sf.loadVars('itr', 'iva')
            yellow_err('BSVM.simple__gridsearch__(): try {0}'.format(sf))
        except:
            while True:
                yellow_err('BSVM.simple__gridsearch__(): except {0}'.format(sf))
                itr, iva = mlpy.kfold(len(self._vftr), 5, rseed=random.random())[0]
                if len(iva) == 1: iva.append(itr.pop())
                if (
                    len(set([self._vltr[i] for i in itr])) == 2 and
                    len(set([self._vltr[i] for i in iva])) == 2
                    ):
                    break
            sf.saveVars(itr=itr, iva=iva)

        """
        vffi = vector of fitting features
        vlfi = vector of fitting labels
        vfva = vector of validation features
        vlva = vector of validation labels
        """
        vffi, vlfi = (np.array(self._vftr)[itr]).tolist(), (np.array(self._vltr)[itr]).tolist()
        vfva, vlva = (np.array(self._vftr)[iva]).tolist(), (np.array(self._vltr)[iva]).tolist()

        """
        Printing
        """
        yellow_err('BSVM.simple__gridsearch__(): self._acs: {0}'.format(self._acs))
        vlfi2, vlva2 = vlfi[:], vlva[:]
        for c in self._acs + [self._unknown_label]:
            assert isinstance(vlfi2, list)
            assert isinstance(vlva2, list)
            yellow_err('BSVM.simple__gridsearch__(): {0} {1} {2}'.format(c, vlfi2.count(c), vlva2.count(c)))

        def ofBSVM(c, g):
            """
            Objective Function.
            """
            classifier = SVM(training_function=self._training_function,
                             prediction_function=self._prediction_function,
                             C=c,
                             gamma=g)
            classifier.fit(vffi, vlfi)
            vlpr = classifier.classify(vfva, vlva)

            """
            One of the two labels is considered unknown for
            result acquisition.
            """
            aux_unknown_label = min(self._acs)

            results = common.predictionResults(
                [x if x != aux_unknown_label else None
                 for x in vlva],
                [x if x != aux_unknown_label else None
                 for x in vlpr],
                aux_unknown_label,
                have_unknown=True,
                )
            ret = results['NA']

            yellow_err('BSVM.simple__gridsearch__(): gamma = {0} result: {1} = ({2} + {3})/2'
                       .format(g, ret, results['AKS'], results['AUS']))
            return ret

        params = common.gridSearchExponential('min', ofBSVM,
                                              [2**i for i in range(-5, 15 + 1)],
                                              [2**i for i in range(-15, 3 + 1)])
        c, g = params

        assert not self._parameters.has_key('C')
        assert not self._parameters.has_key('gamma')

        self._parameters['C'] = c
        self._parameters['gamma'] = g

        yellow_err('BSVM.simple__gridsearch__(): C, gamma = {0}, {1}'.format(c, g))

    def simplefit(self):
        assert len(self._acs) == 2
        SVM.simplefit(self)

    def simpleclassify(self, auxiliar=None):
        ret = SVM.simpleclassify(self, auxiliar)

        if auxiliar is not None:
            assert all([len(x) <= 2
                        for x in auxiliar])

        return ret


class SVMDBC(BSVM):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('training_function')
        assert not self._parameters.has_key('prediction_function')

        self._parameters['training_function'] = svm_dbc_training_function
        self._parameters['prediction_function'] = svm_prediction_function

        SVM.simple__init__(self)

        if self._parameters.has_key('C') and self._parameters.has_key('gamma'):
            self._gridsearch = False
        else:
#             assert not self._parameters.has_key('C')
#             assert not self._parameters.has_key('gamma')
            self._gridsearch = True

    def simple__repr__(self):
        return 'SVMDBC'


class SVMSH(BSVM):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('training_function')
        assert not self._parameters.has_key('prediction_function')

        if self._parameters.has_key('positive_label'):
            assert isinstance(self._parameters['positive_label'], int)
        else:
            self._parameters['positive_label'] = None

        if self._parameters.has_key('num_classifiers'):
            assert isinstance(self._parameters['num_classifiers'], int)
        else:
            self._parameters['num_classifiers'] = 0

        self._parameters['training_function'] = svm_sh_training_function
        self._parameters['prediction_function'] = svm_sh_prediction_function

        SVM.simple__init__(self)

        self._gridsearch = False

    def simplefit(self):
        assert self._parameters['positive_label'] is not None

        self.adjust_labels()
        # SVM.simplefit(self) below, but with another parameter to the training function
        Classifier.simplefit(self)

        vftr = map(SVM._featureVectorFromListToDict, self._vftr)

        vltr = [label
                if label is not None
                else self._unknown_label
                for label in self._vltr]
        str_parameters = ((' -c {0} '.format(self._parameters['C'])
                           if self._parameters.has_key('C')
                           else '') +
                          (' -g {0} '.format(self._parameters['gamma'])
                           if self._parameters.has_key('gamma')
                           else ''))
        self._model = self._training_function(vltr, vftr, str_parameters, self._parameters['num_classifiers'])

    def simpleconfig(self):
        assert isinstance(self._parameters['positive_label'], int)
        assert isinstance(self._parameters['num_classifiers'], int)

    def simpleclassify(self, auxiliar=None):
        Classifier.simpleclassify(self, auxiliar)

        vfte = map(SVM._featureVectorFromListToDict, self._vfte)

        vlte = [label
                if label is not None
                else self._unknown_label
                for label in self._vlte]
        vlpr, apr, vpr = self._prediction_function(vlte, vfte, self._model, ' -q ')
        vlpr = [label
                if label != self._unknown_label and label != -1
                else None # Put none whenever returns unknown or -1 (which is unknown too in SVMSH)
                for label in vlpr]

        if auxiliar is not None:
            for x in vpr: # this for could be replaced by auxiliar.extend(vpr)
                auxiliar.append(x)

        assert len(vlpr) == len(vfte)

        del vfte
        del vlte

        return vlpr

    def simple__repr__(self):
        return 'SVMSH'

    def adjust_labels(self):
        """
        """
        lls = self._vltr
        lfs = self._vftr

        assert len(lls) == len(lfs) or len(lfs) == 0
        assert self._parameters['positive_label'] in lls

        lls = [x if x == self._parameters['positive_label'] else -x for x in lls]
        idx = lls.index(self._parameters['positive_label'])
        lls = lls[idx:] + lls[:idx]
        lfs = lfs[idx:] + lfs[:idx]

        assert next(len(x) == 1 and x.pop() == self._parameters['positive_label'] for x in [set([y for y in lls if y > 0])])

        self._vltr = lls
        self._vftr = lfs


class MCBBClassifier(Classifier):
    """
    Multiclass binary-based classifier.
    """
    def simple__init__(self):
        """
        bc: binary classifier
        """
        Classifier.simple__init__(self)

        assert self._parameters.has_key('bc')

        self._bc = self._parameters['bc']
        self._approach = (self._parameters['approach']
                          if self._parameters.has_key('approach')
                          else 'OVA')
        """
        self._ovaonlyone - OVA only one - this variable indicates if
        one and only one binary classifier must classify as positive
        to the general MCBB classifier classify as one of the
        available classes.  Otherwise, the MCBB classifier are going
        to classify as unknown (if self._ovaonlyone is True).
        """
        self._ovaonlyone = (self._parameters['ovaonlyone']
                            if self._parameters.has_key('ovaonlyone')
                            else False)

        self._gridsearch = False

        assert self._approach in ['OVO', 'OVA']

    def simple__repr__(self):
        return 'MCBBClassifier'

    def simplefit(self):
        Classifier.simplefit(self)

        assert self._bc
        assert self._unknown_label not in self._vltr

        self._binary_classifiers = []

        if self._approach == 'OVO':
            for i in range(0, len(self._acs)):
                for j in range(i + 1, len(self._acs)):
                    vl, vf = common.limitTraining(self._vltr,
                                                  self._vftr,
                                                  [self._acs[i],
                                                   self._acs[j]])

                    bc = self._bc.copy()
                    bc.fit(vf, vl)

                    self._binary_classifiers.append(bc)

        elif self._approach == 'OVA':
            for clss in self._acs:
                vl = [l
                      if l == clss
                      else self._unknown_label
                      for l in self._vltr]
                vf = self._vftr[:]

                if (vl.count(self._unknown_label) > 0 and
                    vl[0] != self._unknown_label):
                    """
                    Making all negative samples be first on the list
                    of training samples.  The SVM1VS treats
                    differently depending on the order of the samples.

                    pedrormjunior 20131029
                    """
                    idx = vl.index(self._unknown_label)
                    vl = vl[idx:] + vl[:idx]
                    vf = vf[idx:] + vf[:idx]

                bc = self._bc.copy()
                bc.fit(vf, vl)

                self._binary_classifiers.append(bc)

        else:
            raise ValueError('Unrecognized approach for multiclass classification.')

    def simpleclassify(self, auxiliar=None):
        Classifier.simpleclassify(self, auxiliar)

        lst_vlpr, lst_margin_distances = [], []
        for bc in self._binary_classifiers:
            margin_distances = []
            vlpr = bc.classify(self._vfte,
                               self._vlte,
                               margin_distances)
            vlpr = [label
                    if label != self._unknown_label
                    else None
                    for label in vlpr]

            if auxiliar is not None:
                """
                The `auxiliar` contains the information of the
                classification of each binary classifier.

                For now, I am aiming to use this information to
                calculate the Binary-based F-measure.

                pedrormjunior 20131030
                """
                auxiliar.append(vlpr[:])

            else:
                magenta_err('MCBBClassifier.simpleclassify(): auxiliar is None')

            lst_vlpr.append(vlpr);
            if self._ovaonlyone:
                lst_margin_distances.append([x[:]
                                             for x in margin_distances])
            else:
                lst_margin_distances.append([x[0]
                                             for x in margin_distances])

        class caux:
            def __init__(self, data):
                self.data = data

            def __repr__(self):
                return 'caux({0})'.format(self.data)

        lst_vlpr_mdists = map(lambda (lst1, lst2): map(caux, zip(lst1, lst2)), zip(lst_vlpr, lst_margin_distances))
        lst_vlpr_mdists = np.array(lst_vlpr_mdists).transpose().tolist()
        lst_vlpr_mdists = map(lambda x: map(lambda y: y.data,
                                       x),
                              lst_vlpr_mdists)

        if self._approach == 'OVO':
            if True:
                def decisionFunction(lst):
                    lst = map(lambda (clss, mdist): clss, lst)

                    _, clss = max([(lst.count(clss), clss)
                                   for clss
                                   in self._acs])

                    return clss

            else:
                def decisionFunction(lst):
                    clss, _ = max(lst,
                                  key=lambda (clss, mdist): abs(mdist))

                    return clss

        elif self._approach == 'OVA':
            if self._ovaonlyone:
                def decisionFunction(lst):
                    """
                    If only one binary classifier classifies as
                    positive, then its positive label is the final
                    decision of the general MCBB classifier.  If no
                    binary classifier classify as positive, the final
                    label is unknown.  If two or more binary
                    classifier classify as positive, the final label
                    is unknown too.
                    """
                    clss_mdist = lst[:]
                    lst = filter(lambda (clss, mdist): clss != None, lst)

                    if len(lst) == 1:
                        if (all(map(lambda (clss, mdist): len(mdist) == 2, clss_mdist)) and
                            any(map(lambda (clss, mdist): mdist[1] < 0, clss_mdist))):
                            """
                            It accounts for a special case on the
                            MCSVM1VS_ovaonlyone that is the case when
                            the test sample is on the positive region
                            of only one binary SVM1VS but is faraway
                            from the omega hyperplane of some other
                            SVM1VS by the positive side.

                            A test sample on the MCSVM1VS_ovaonlyone
                            are going to be classified as known only
                            if one, and only one, SVM1VS classify as
                            positive with the aditional constraint
                            that the test sample must the nearest to
                            the alpha hyperplane, instead of the
                            omega, of the other SVM1VSs.
                            """
                            clss = None
                        else:
                            (clss, _), = lst
                    else:
                        clss = None

                    return clss

            else:
                if True:
                    def decisionFunction(lst):
                        """
                        Among the positive classifications, choose
                        positive class classified with maximum distance
                        from margin.  It avoids a random choice.
                        """
                        lst = filter(lambda (clss, mdist): clss != None, lst)
#                         print lst

                        if lst == []:
                            clss = None
                        else:
                            clss, _ = max(lst,
                                          key=lambda (clss, mdist): abs(mdist))

                        return clss

                else:
                    def decisionFunction(lst):
                        """
                        Among the positive classifications, choose
                        positive class classified with maximum distance
                        from margin.  It avoids a random choice.
                        """
                        lst = filter(lambda (clss, mdist): clss != None, lst)

#                         green_err(lst)
                        if lst == []:
                            clss = None
                        else:
                            clss, _ = min(lst,
                                          key=lambda (clss, mdist): clss)

                        return clss

        else:
            raise ValueError('Unrecognized approach for multiclass classification.')

        vlpr = map(decisionFunction, lst_vlpr_mdists)

        assert len(vlpr) == len(self._vfte)

        return vlpr

    def pls(self):
        ret = reduce(lambda a, b: a + b,
                     [bc.pls()
                      for bc in self._binary_classifiers])
        while True:
            try:
                ret.remove(self._unknown_label)
            except:
                break

        return ret


class MCSVM(MCBBClassifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('bc')

        self._parameters['bc'] = BSVM()

        MCBBClassifier.simple__init__(self)

        self._gridsearch = False

    def simple__repr__(self):
        if self._approach == 'OVA' and self._ovaonlyone:
            ret = 'MCSVM_ovaonlyone'
        elif self._approach == 'OVO' and not self._ovaonlyone:
            ret = 'MCSVM_OVO'
        elif self._approach == 'OVO' and self._ovaonlyone:
            raise ValueError('It does not make sense (20131028)')
        else:
            ret = 'MCSVM'
        return ret


class MCSVMexternal(MCBBClassifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('bc')

        if self._parameters.has_key('C') and self._parameters.has_key('gamma'):
            self._parameters['bc'] = BSVM(C=self._parameters['C'], gamma=self._parameters['gamma'])
        else:
            self._parameters['bc'] = BSVM(C=1.0, gamma=0.1)

        MCBBClassifier.simple__init__(self)

        if self._parameters.has_key('C') and self._parameters.has_key('gamma'):
            self._gridsearch = False
        else:
            self._gridsearch = True

    def simple__repr__(self):
        return 'MCSVMexternal'

    def simple__gridsearch__(self):
        Classifier.simple__gridsearch__(self)

        #Werneck
        #hf = current_simulations_settings['hf']
        hash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage"))
        if not os.path.exists(hash_path):
            os.mkdir(hash_path)
        hf = hashfile.HashFile(hash_path + os.sep, 'HashFile.npz')
        #-------
        sf = hf.getStorageFile([self._acs, self._vftr, self._vltr])

        try:
            itr, iva = sf.loadVars('itr', 'iva')
            yellow_err('MCSVMexternal.simple__gridsearch__(): try {0}'.format(sf))
        except:
            while True:
                yellow_err('MCSVMexternal.simple__gridsearch__(): except {0}'.format(sf))
                itr, iva = mlpy.kfold(len(self._vftr), 5, rseed=random.random())[0]
                if len(iva) == 1: iva.append(itr.pop())
                if (
#                     len(set([self._vltr[i] for i in itr])) == len(set(self._acs)) and
#                     len(set([self._vltr[i] for i in iva])) == len(set(self._acs))
                    len(set([self._vltr[i] for i in itr])) >= 2 and
                    len(set([self._vltr[i] for i in iva])) >= 2
                    ):
                    break
            sf.saveVars(itr=itr, iva=iva)

        """
        vffi = vector of fitting features
        vlfi = vector of fitting labels
        vfva = vector of validation features
        vlva = vector of validation labels
        """
        vffi, vlfi = (np.array(self._vftr)[itr]).tolist(), (np.array(self._vltr)[itr]).tolist()
        vfva, vlva = (np.array(self._vftr)[iva]).tolist(), (np.array(self._vltr)[iva]).tolist()

        """
        Printing
        """
        yellow_err('MCSVMexternal.simple__gridsearch__(): self._acs: {0}'.format(self._acs))
        vlfi2, vlva2 = vlfi[:], vlva[:]
        for c in self._acs + [self._unknown_label]:
            assert isinstance(vlfi2, list)
            assert isinstance(vlva2, list)
            yellow_err('MCSVMexternal.simple__gridsearch__(): {0} {1} {2}'.format(c, vlfi2.count(c), vlva2.count(c)))

        def ofMCSVMexternal(c, g):
            """
            Objective Function.
            """
            classifier = MCSVMexternal(C=c,
                                       gamma=g)
            classifier.fit(vffi, vlfi)
            vlpr = classifier.classify(vfva, [None]*len(vlva))

            """
            One of the two labels is considered unknown for
            result acquisition.
            """
            aux_unknown_label = min(self._acs)

            results = common.predictionResults(
                [x if x != aux_unknown_label else None
                 for x in vlva],
                [x if x != aux_unknown_label else None
                 for x in vlpr],
                aux_unknown_label,
                have_unknown=True,)
            ret = results['NA']

            yellow_err('MCSVMexternal.simple__gridsearch__(): gamma = {0} result: {1} = ({2} + {3})/2'
                       .format(g, ret, results['AKS'], results['AUS']))
            return ret

        params = common.gridSearchExponential('min', ofMCSVMexternal,
                                              [2**i for i in range(-5, 15 + 1)],
                                              [2**i for i in range(-15, 3 + 1)])
        c, g = params

        assert not self._parameters.has_key('C')
        assert not self._parameters.has_key('gamma')

        self._parameters['C'] = c
        self._parameters['gamma'] = g

        self._bc.config(C=self._parameters['C'],
                        gamma=self._parameters['gamma'])

        yellow_err('MCSVMexternal.simple__gridsearch__(): C, gamma = {0}, {1}'.format(c, g))


class MCSVMDBC(MCBBClassifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('bc')

        self._parameters['bc'] = SVMDBC()

        MCBBClassifier.simple__init__(self)

        self._gridsearch = False

    def simple__repr__(self):
        return 'MCSVMDBC'


class MCSVMSH(MCBBClassifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('bc')

        self._parameters['bc'] = SVMSH()

        if self._parameters.has_key('num_classifiers'):
            assert isinstance(self._parameters['num_classifiers'], int)
        else:
            self._parameters['num_classifiers'] = None

        MCBBClassifier.simple__init__(self)

        self._gridsearch = False

    def simplefit(self):
        Classifier.simplefit(self)

        assert self._bc
        assert self._unknown_label not in self._vltr

        self._binary_classifiers = []

        if self._parameters['num_classifiers'] is None:
            self._parameters['num_classifiers'] = ((self._acs - 1) * 2) - 1 # 2n-1 where n is nr of negative classes

        for clss in self._acs:
            bc = self._bc.copy()
            bc.config(positive_label=clss, num_classifiers = self._parameters['num_classifiers'])
            bc.fit(self._vftr, self._vltr)

            self._binary_classifiers.append(bc)

    def simple__repr__(self):
        return 'MCSVMSH'

    def simpleclassify(self, auxiliar=None):
        Classifier.simpleclassify(self, auxiliar)

        lst_vlpr, lst_margin_distances = [], []
        for bc in self._binary_classifiers:
            margin_distances = []
            vlpr = bc.classify(self._vfte,
                               self._vlte,
                               margin_distances)
            vlpr = [label
                    if label != self._unknown_label
                    else None
                    for label in vlpr]

            lst_vlpr.append(vlpr);
            lst_margin_distances.append([x[0]
                                         for x in margin_distances])

        class caux:
            def __init__(self, data):
                self.data = data

            def __repr__(self):
                return 'caux({0})'.format(self.data)

        lst_vlpr_mdists = map(lambda (lst1, lst2): map(caux, zip(lst1, lst2)), zip(lst_vlpr, lst_margin_distances))
        lst_vlpr_mdists = np.array(lst_vlpr_mdists).transpose().tolist()
        lst_vlpr_mdists = map(lambda x: map(lambda y: y.data,
                                       x),
                              lst_vlpr_mdists)

        if False:
            def decisionFunction(lst):
                """
                Among the positive classifications, choose
                positive class most frequent.  It does not
                make sense because there is at most one
                positive class classification for a specified
                class in the OVA approach.
                """
                lst = map(lambda (clss, mdist): clss, lst)
                lst = filter(lambda clss: clss != None, lst)

                if lst == []:
                    clss = None
                else:
                    _, clss = max([(lst.count(clss), clss)
                                   for clss
                                   in self._acs])

                return clss
        else:
            def decisionFunction(lst):
                """
                Among the positive classifications, choose
                positive class classified with maximum
                distance from margin.  It garanties the
                sample is correctly classified.
                """
                lst = filter(lambda (clss, mdist): clss != None, lst)

                if lst == []:
                    clss = None
                else:
                    clss, _ = max(lst,
                                  key=lambda (clss, mdist): abs(mdist))

                return clss

        vlpr = map(decisionFunction, lst_vlpr_mdists)

        assert len(vlpr) == len(self._vfte)

        return vlpr


class OCSVM(SVM):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('training_function')
        assert not self._parameters.has_key('prediction_function')

        self._parameters['training_function'] = one_class_svm_training_function
        self._parameters['prediction_function'] = svm_prediction_function

        SVM.simple__init__(self)

        self._gridsearch = (False
                            if self._parameters.has_key('gamma')
                            else True)

    def simple__repr__(self):
        return 'OCSVM'

    def simple__gridsearch__(self):
        Classifier.simple__gridsearch__(self)

        ovltr, ovftr = (map(list,
                            zip(*filter(lambda (l, f): l < 0,
                                        zip(self._vltr, self._vftr))))
                        if len(filter(lambda x: x < 0,
                                      self._vltr)) >= 1
                        else [[], []])
        ovltr = map(lambda _: None,
                    ovltr)

        self._vltr, self._vftr = map(list,
                                     zip(*filter(lambda (l, f): l > 0,
                                                 zip(self._vltr, self._vftr))))

        #Werneck
        #hf = current_simulations_settings['hf']
        hash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage"))
        if not os.path.exists(hash_path):
            os.mkdir(hash_path)
        hf = hashfile.HashFile(hash_path + os.sep, 'HashFile.npz')
        #-------
        sf = hf.getStorageFile([self._acs, self._vftr, self._vltr])

        try:
            itr, iva = sf.loadVars('itr', 'iva')
            yellow_err('OCSVM.simple__gridsearch__(): try')
        except:
            yellow_err('OCSVM.simple__gridsearch__(): except')
            itr, iva = mlpy.kfold(len(self._vftr), 5, rseed=random.random())[0]
            sf.saveVars(itr=itr, iva=iva)

        """
        vffi = vector of fitting features
        vlfi = vector of fitting labels
        vfva = vector of validation features
        vlva = vector of validation labels
        """
        vffi, vlfi = (np.array(self._vftr)[itr]), (np.array(self._vltr)[itr])
        vfva, vlva = (np.array(self._vftr)[iva]).tolist(), (np.array(self._vltr)[iva]).tolist()

        """
        Printing
        """
        yellow_err('OCSVM.simple__gridsearch__(): self._acs: {0}'.format(self._acs))
        vlfi2, vlva2 = vlfi.tolist(), vlva[:]
        for c in self._acs + [self._unknown_label]:
            assert isinstance(vlfi2, list)
            assert isinstance(vlva2, list)
            yellow_err('OCSVM.simple__gridsearch__(): {0} {1} {2}'.format(c, vlfi2.count(c), vlva2.count(c)))

        def ofOCSVM(g):
            """
            Objective Function.
            """
            classifier = OCSVM(gamma=g)
            classifier.fit(vffi, vlfi)
            vlpr = classifier.classify(vfva + ovftr,
                                       vlva + ovltr,
                                       )

            """
            PEDRORMJUNIOR: the case in which all predicted labels
            are non-positive, it will raise a problem inside the
            'predictionResults()' function.  I must change the
            'predictionResults()' function.
            """
            results = common.predictionResults(vlva + ovltr,
                                               vlpr,
                                               self._unknown_label,
                                               have_unknown=True,
                                               )
#             ret = results['AKS'] # accuracy of known samples
            ret = results['NA']

            yellow_err('OCSVM.simple__gridsearch__(): gamma = {0} result: {1}'.format(g, ret))
            return ret

        params = common.gridSearchExponential('min', ofOCSVM, [2**i for i in range(-15, 3 + 1)])
        g, = params

        self._parameters['gamma'] = g

        yellow_err('OCSVM.simple__gridsearch__(): gamma = {0}'.format(g))

    def simplefit(self):
        """
        You can give more than one label to fit a OCSVM.
        But only the samples with positive label are going to be used fot really fit the classifier.
        In this vein, the set of positive labels must be 1, i.e., only one label n such than n > 0 must be given.
        """
        assert len(set(self._vltr)) == 1
        SVM.simplefit(self)

    def simpleclassify(self, auxiliar=None):
        positive_label = max(self._acs)

        ret = SVM.simpleclassify(self, auxiliar)
        ret = [None
               if x == -1
               else positive_label
               for x in ret]

        return ret


class MCOCBClassifier(Classifier):
    """
    Multiclass one-class--based classifier.
    """
    def simple__init__(self):
        """
        occ: one-class classifier
        """
        Classifier.simple__init__(self)

        assert self._parameters.has_key('occ')
        if self._parameters.has_key('allsamplesgs'): # all samples in grid search
            assert isinstance(self._parameters['allsamplesgs'], bool)
            self._allsamplesgs = self._parameters['allsamplesgs']
        else:
            self._allsamplesgs = False

        self._occ = self._parameters['occ']

        self._gridsearch = False

    def simple__repr__(self):
        return 'MCOCBClassifier'

    def simplefit(self):
        Classifier.simplefit(self)

        assert hasattr(self, '_occ')
        assert self._unknown_label not in self._vltr

        self._one_class_classifiers = []

        for clss in self._acs:
            if self._allsamplesgs:
                vl = map(lambda l: (l if l == clss else -l),
                         self._vltr)

                occ = self._occ.copy()
                occ.fit(self._vftr, vl)

            else:
                vf, vl = map(list,
                             zip(*[(f, l)
                                   for f, l in zip(self._vftr, self._vltr)
                                   if l == clss]))

                occ = self._occ.copy()
                occ.fit(vf, vl)

            self._one_class_classifiers.append(occ)

    def simpleclassify(self, auxiliar=None):
        Classifier.simpleclassify(self, auxiliar)

        lst_vlpr, lst_margin_distances = [], []
        for occ in self._one_class_classifiers:
            margin_distances = []
            vlpr = occ.classify(self._vfte,
                                self._vlte,
                                margin_distances)

            lst_vlpr.append(vlpr);
            lst_margin_distances.append([x[0]
                                         for x in margin_distances])

        class caux:
            def __init__(self, data):
                self.data = data

            def __repr__(self):
                return 'caux({0})'.format(self.data)

        lst_vlpr_mdists = map(lambda (lst1, lst2): map(caux, zip(lst1, lst2)), zip(lst_vlpr, lst_margin_distances))
        lst_vlpr_mdists = np.array(lst_vlpr_mdists).transpose().tolist()
        lst_vlpr_mdists = map(lambda x: map(lambda y: y.data,
                                       x),
                              lst_vlpr_mdists)

        if False:
            def decisionFunction(lst):
                """
                Among the positive classifications, choose
                positive class most frequent.  It does not make
                sense because there is at most one positive class
                classification for a specified class in the OVA
                approach.
                """
                lst = map(lambda (clss, mdist): clss, lst)
                lst = filter(lambda clss: clss != None, lst)

                if lst == []:
                    clss = None
                else:
                    _, clss = max([(lst.count(clss), clss)
                                   for clss
                                   in self._acs])

                return clss
        else:
            def decisionFunction(lst):
                """
                Among the positive classifications, choose
                positive class classified with maximum distance
                from margin.  It garanties the sample is
                correctly classified.
                """
                lst = filter(lambda (clss, mdist): clss != None, lst)

                if lst == []:
                    clss = None
                else:
                    clss, _ = max(lst,
                                  key=lambda (clss, mdist): abs(mdist))

                return clss

        vlpr = map(decisionFunction, lst_vlpr_mdists)

        assert len(vlpr) == len(self._vfte)

        return vlpr

    def pls(self):
        ret = reduce(lambda a, b: a + b,
                     [occ.pls()
                      for occ in self._one_class_classifiers])
        while True:
            try:
                ret.remove(self._unknown_label)
            except:
                break

        return ret


class MCOCSVM(MCOCBClassifier):
    """
    Multiclass one-class SVM.
    """
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('occ')

        self._parameters['occ'] = OCSVM()

        MCOCBClassifier.simple__init__(self)

        self._gridsearch = False

    def simple__repr__(self):
        if self._allsamplesgs:
            ret = 'MCOCSVM_allsamplesgs'
        else:
            ret = 'MCOCSVM'
        return ret


class SVM1VS(BSVM):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('training_function')
        assert not self._parameters.has_key('prediction_function')

        self._parameters['training_function'] = svm_1vs_training_function
        self._parameters['prediction_function'] = svm_1vs_prediction_function

        if not self._parameters.has_key('kernel_type'):
            self._parameters['kernel_type'] = 0

        SVM.simple__init__(self)

        if (self._parameters.has_key('C') and
            self._parameters.has_key('near_pressure') and
            self._parameters.has_key('far_pressure')):
#         if (self._parameters.has_key('C')):
            self._gridsearch = False
        else:
            self._gridsearch = True

    def simple__gridsearch__(self):
        Classifier.simple__gridsearch__(self)

        #Werneck
        #hf = current_simulations_settings['hf']
        hash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage"))
        if not os.path.exists(hash_path):
            os.mkdir(hash_path)
        hf = hashfile.HashFile(hash_path + os.sep, 'HashFile.npz')
        #-------
        sf = hf.getStorageFile([self._acs, self._vftr, self._vltr])

        try:
            itr, iva = sf.loadVars('itr', 'iva')
            yellow_err('SVM1VS.simple__gridsearch__(): try {0}'.format(sf))
        except:
            while True:
                yellow_err('SVM1VS.simple__gridsearch__(): except {0}'.format(sf))
                itr, iva = mlpy.kfold(len(self._vftr), 5, rseed=random.random())[0]
                if len(iva) == 1: iva.append(itr.pop())
                if (
                    len(set([self._vltr[i] for i in itr])) == 2 and
                    len(set([self._vltr[i] for i in iva])) == 2
                    ):
                    break
            sf.saveVars(itr=itr, iva=iva)

        """
        vffi = vector of fitting features
        vlfi = vector of fitting labels
        vfva = vector of validation features
        vlva = vector of validation labels
        """
        vffi, vlfi = (np.array(self._vftr)[itr]).tolist(), (np.array(self._vltr)[itr]).tolist()
        vfva, vlva = (np.array(self._vftr)[iva]).tolist(), (np.array(self._vltr)[iva]).tolist()

        """
        Printing
        """
        yellow_err('SVM1VS.simple__gridsearch__(): self._acs: {0}'.format(self._acs))
        vlfi2, vlva2 = vlfi[:], vlva[:]
        for c in self._acs + [self._unknown_label]:
            assert isinstance(vlfi2, list)
            assert isinstance(vlva2, list)
            yellow_err('SVM1VS.simple__gridsearch__(): {0} {1} {2}'.format(c, vlfi2.count(c), vlva2.count(c)))

        if self._parameters['kernel_type'] == 2:
            def ofSVM1VS(c, g, near_pressure, far_pressure):
                """
                Objective Function.
                """
                classifier = SVM1VS(
                    kernel_type=2,
                    C=c,
                    gamma=g,
                    near_pressure=near_pressure,
                    far_pressure=far_pressure,
                    )
                classifier.fit(vffi, vlfi)
                vlpr = classifier.classify(vfva, vlva)

                """
                One of the two labels is considered unknown for
                result acquisition.
                """
                aux_unknown_label = min(self._acs)

                results = common.predictionResults(
                    [x if x != aux_unknown_label else None
                     for x in vlva],
                    [x if x != aux_unknown_label else None
                     for x in vlpr],
                    aux_unknown_label,
                    have_unknown=True,
                    )
                ret = results['NA']

                yellow_err('SVM1VS.simple__gridsearch__(): C, gamma, near, far = {0} {1} {2} {3} result: {4} = ({5} + {6})/2'
                           .format(c, g, near_pressure, far_pressure, ret, results['AKS'], results['AUS']))
                return ret

            params = common.gridSearchExponential('min', ofSVM1VS,
                                                  [2**i for i in range(-5, 15 + 1)],
                                                  [2**i for i in range(-15, 3 + 1)],
                                                  [2**i for i in [-15, -7, -2, 0, 2]],
                                                  [2**i for i in [-15, -7, -2, 0, 2]],
                                                  )
            c, g, near_pressure, far_pressure = params

            assert not self._parameters.has_key('C')
            assert not self._parameters.has_key('gamma')
            assert not self._parameters.has_key('near_pressure')
            assert not self._parameters.has_key('far_pressure')

            self._parameters['C'] = c
            self._parameters['gamma'] = g
            self._parameters['near_pressure'] = near_pressure
            self._parameters['far_pressure'] = far_pressure

            yellow_err('SVM1VS.simple__gridsearch__(): C, gamma, near, far = {0}, {1}, {2}, {3}'.format(c, g, near_pressure, far_pressure))

        elif self._parameters['kernel_type'] == 0:
            def ofSVM1VS(c, near_pressure, far_pressure):
                """
                Objective Function.
                """
                classifier = SVM1VS(
                    kernel_type=0,
                    C=c,
                    near_pressure=near_pressure,
                    far_pressure=far_pressure,
                    )
                classifier.fit(vffi, vlfi)
                vlpr = classifier.classify(vfva, vlva)

                """
                One of the two labels is considered unknown for
                result acquisition.
                """
                aux_unknown_label = min(self._acs)

                results = common.predictionResults(
                    [x if x != aux_unknown_label else None
                     for x in vlva],
                    [x if x != aux_unknown_label else None
                     for x in vlpr],
                    aux_unknown_label,
                    have_unknown=True,
                    )
                ret = results['NA']

                yellow_err('SVM1VS.simple__gridsearch__(): C, near, far = {0}, {1}, {2} result: {3} = ({4} + {5})/2'
                           .format(c, near_pressure, far_pressure, ret, results['AKS'], results['AUS']))
                return ret

            params = common.gridSearchExponential('min', ofSVM1VS,
                                                  [2**i for i in range(-5, 15 + 1)],
                                                  [2**i for i in [-15, -7, -2, 0, 2]],
                                                  [2**i for i in [-15, -7, -2, 0, 2]],
                                                  )
            c, near_pressure, far_pressure = params

            assert not self._parameters.has_key('C')
            assert not self._parameters.has_key('near_pressure')
            assert not self._parameters.has_key('far_pressure')

            self._parameters['C'] = c
            self._parameters['near_pressure'] = near_pressure
            self._parameters['far_pressure'] = far_pressure

            yellow_err('SVM1VS.simple__gridsearch__(): C, near, far = {0} {1} {2}'.format(c, near_pressure, far_pressure))

        else:
            raise ValueError('Unrecognized kernel_type for SVM1VS.')

    def simplefit(self):
        SVM.simplefit(self)

    def simple__repr__(self):
        return 'SVM1VS'


class MCSVM1VS(MCBBClassifier):
    def simple__init__(self):
        Classifier.simple__init__(self)

        assert not self._parameters.has_key('bc')

        if not self._parameters.has_key('kernel_type'):
            self._parameters['kernel_type'] = 0

        self._parameters['bc'] = SVM1VS(
            kernel_type=self._parameters['kernel_type'],
            )

        MCBBClassifier.simple__init__(self)

        self._gridsearch = False

    def simple__repr__(self):
        def kerneltype():
            if self._parameters['kernel_type'] == 0:
                return ''
            elif self._parameters['kernel_type'] == 2:
                return '_rbf'
            else:
                return '_unknownkerneltype'

        def ovaonlyone():
            if self._ovaonlyone:
                return '_ovaonlyone'
            else:
                return ''

        return 'MCSVM1VS' + kerneltype() + ovaonlyone()
