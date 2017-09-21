#!/usr/bin/env bash

path_to_libsvm=../libraries/libSVM-onevset/

if [ "$1" == "clean" ]; then
    rm ../lib1vs.so.2
    rm ./svm1vs.py
    rm ./svmutil1vs.py
else
    ln -s ${path_to_libsvm}lib1vs.so.2 ../lib1vs.so.2
    ln -s ../${path_to_libsvm}python/svm1vs.py svm1vs.py
    ln -s ../${path_to_libsvm}python/svmutil1vs.py svmutil1vs.py
fi
