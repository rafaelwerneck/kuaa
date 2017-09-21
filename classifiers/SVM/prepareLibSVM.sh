#!/usr/bin/env bash

path_to_libsvm=../libraries/libsvm-3.17/

if [ "$1" == "clean" ]; then
    rm ../libsvm.so.2
    rm ./svm.py
    rm ./svmutil.py
else
    ln -s ${path_to_libsvm}libsvm.so.2 ..
    ln -s ../${path_to_libsvm}python/svm.py .
    ln -s ../${path_to_libsvm}python/svmutil.py .
fi
