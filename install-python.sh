#!/bin/bash -e

cd /tmp

echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

apt-get update
apt-get install -y --no-install-recommends apt-utils
apt-get install -y --no-install-recommends build-essential cmake git mercurial bzr pkg-config vim
apt-get install -y --no-install-recommends wget curl unzip tar
apt-get install -y --no-install-recommends libffi-dev

pip install --upgrade pip
#pip install h5py
#pip install PyOpenGL
#pip install Pillow
#pip install cython
#pip install numpy
#pip install scipy
#pip install scikit-image
#pip install scikit-learn
#pip install pandas
#pip install matplotlib
#pip install seaborn
#pip install bokeh
#pip install plotly
#pip install jupyter
pip install enum34 
pip install future
#pip install visdom dominate
pip install setuptools
#pip install protobuf
pip install opencv-python

rm -rf /var/lib/apt/lists/*



