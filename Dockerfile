FROM ubuntu:16.04

# Install all dependencies
RUN apt-get update && \
	apt-get -qq remove ffmpeg x264 libx264-dev && \
	apt-get install sudo -qqy --no-upgrade \
	wget unzip cmake  \
        libqwt-qt5-6 libqwt-qt5-dev \
	qt5-default libqt5concurrent5 libqt5test5 \
	checkinstall \
	python-tk python-imaging-tk libjpeg62 \
	python-pydot \
	build-essential \
        python-dev \
        python-setuptools \
        python-scipy \
        libatlas-dev \
	python-matplotlib \
	python-pip \
	python-mlpy \
	texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
	openjdk-8-jdk jython jellyfish \
	libopencv-dev build-essential checkinstall cmake \
	pkg-config yasm libjpeg-dev libavcodec-dev \
	libavformat-dev libswscale-dev libdc1394-22-dev \
	libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
	libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev \
	libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev \
	libopencore-amrwb-dev libtheora-dev libvorbis-dev \
	libxvidcore-dev x264 v4l-utils ffmpeg cmake qt5-default checkinstall

	
RUN pip install scikit-learn

# Install OpenCV
RUN mkdir /OpenCV
WORKDIR /OpenCV
RUN wget -q -O - http://sourceforge.net/projects/opencvlibrary/files/opencv-unix | egrep -m1 -o '\"2.4.11' | cut -c2- > /tmp/version
RUN echo wget -O OpenCV-$(cat /tmp/version).zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$(cat /tmp/version)/opencv-"$(cat /tmp/version)".zip/download
RUN wget -O OpenCV-$(cat /tmp/version).zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$(cat /tmp/version)/opencv-"$(cat /tmp/version)".zip/download
RUN unzip OpenCV-$(cat /tmp/version).zip && rm *.zip
WORKDIR opencv-2.4.11
RUN mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -DENABLE_PRECOMPILED_HEADERS=OFF .. && \
	make -j2
RUN cd build && checkinstall
RUN sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig



RUN mkdir /kuaa
ADD classifiers  config.py    evaluation_measures  fusion_methods    initInterface.py \ 		 train_test_methods \
		collections  descriptors  framework initFramework.py  interface        \
		libraries  normalizers  recommendation \
		/kuaa/
WORKDIR /kuaa/

ENTRYPOINT ["python", "initInterface.py"]



