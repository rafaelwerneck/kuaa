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

###########################################################
# Dependencies installer for Kuaa, a Framework of Machine #
#   Learning Experiments                                  #
#                                                         #
# Author: Rafael Werneck                                  #
#                                                         #
# Version: 1.3.1                                          #
###########################################################
#
# Version 1.3.1
#   removed libxine-dev from OpenCV
#
# Version 1.3
#   added OpenCV 2.4.11
#
# Version 1.2
#   added texlive-fonts-recommended
#
# Version 1.1
#   added python-imaging-tk and jellyfish
###########################################################

#python-tk
sudo apt-get -y install python-tk python-imaging-tk libjpeg62

#numpy
sudo apt-get -y install python-numpy

#pydot
sudo apt-get -y install python-pydot

#sklearn
sudo apt-get -y install build-essential \
                        python-dev \
                        python-setuptools \
                        python-scipy \
                        libatlas-dev
sudo apt-get -y install python-matplotlib
sudo apt-get -y install python-pip
sudo pip install scikit-learn

#mlpy
sudo apt-get -y install python-mlpy

#LaTeX
sudo apt-get -y install texlive-latex-base
sudo apt-get -y install texlive-latex-extra
sudo apt-get -y install texlive-fonts-recommended

#JAVA
sudo apt-get -y install openjdk-7-jdk

#Jython
sudo apt-get -y install jython

#jellyfish
sudo apt-get -y install jellyfish

#CMake (for OpenCV)
sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
sudo apt-get update
sudo apt-get -y install cmake

#OpenCV 2.4.11
mkdir $HOME/OpenCV
cd $HOME/OpenCV
version="$(wget -q -O - http://sourceforge.net/projects/opencvlibrary/files/opencv-unix | egrep -m1 -o '\"2.4.11' | cut -c2-)"
echo "Installing OpenCV" $version
echo "Removing any pre-installed ffmpeg and x264"
sudo apt-get -qq remove ffmpeg x264 libx264-dev
echo "Installing Dependenices"
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils ffmpeg cmake qt5-default checkinstall
echo "Downloading OpenCV" $version
wget -O OpenCV-$version.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$version/opencv-"$version".zip/download
echo "Installing OpenCV" $version
unzip OpenCV-$version.zip
cd opencv-$version
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j2
sudo checkinstall
sudo sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
echo "OpenCV" $version "ready to be used"
