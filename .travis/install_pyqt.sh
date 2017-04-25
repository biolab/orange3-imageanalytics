#!/usr/bin/env bash

PYQT=$TRAVIS_BUILD_DIR/pyqt

SIP_VERSION=4.16.9
PYQT_VERSION=4.11.4

PY_VERSION_FULL=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

PY_BIN_DIR=~/virtualenv/python$PY_VERSION_FULL/bin
PY_SITE_PACKAGES_DIR=~/virtualenv/python$PY_VERSION_FULL/lib/python$TRAVIS_PYTHON_VERSION/site-packages
SIP_DIR=~/virtualenv/python$PY_VERSION_FULL/share/sip

echo $PYTHONPATH

if [ ! "$(ls $PYQT)" ]; then
    mkdir -p $PYQT
    cd $PYQT

    wget -O sip.tar.gz http://sourceforge.net/projects/pyqt/files/sip/sip-$SIP_VERSION/sip-$SIP_VERSION.tar.gz
    mkdir -p sip
    tar xzf sip.tar.gz -C sip --strip-component=1

    wget -O PyQt.tar.gz  http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-$PYQT_VERSION/PyQt-x11-gpl-$PYQT_VERSION.tar.gz
    mkdir -p PyQt
    tar xzf PyQt.tar.gz -C PyQt --strip-components=1

    cd $PYQT/sip
    python configure.py -e $PYQT/include --bindir=$PY_BIN_DIR --destdir=$PY_SITE_PACKAGES_DIR --sipdir=$SIP_DIR
    make
    make install

    cd $PYQT/PyQt
    pwd
    python configure.py --confirm-license --no-designer-plugin --bindir=$PY_BIN_DIR --destdir=$PY_SITE_PACKAGES_DIR --sipdir=$SIP_DIR
    make
fi

cd $PYQT/sip
make install

cd $PYQT/PyQt
make install

cd ../..

pip install pyqtgraph
