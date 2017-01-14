#!/usr/bin/env bash

cd $TRAVIS_BUILD_DIR/orange
# clone orange from git

rm -f -R orange3 # remove if exist
git clone https://github.com/biolab/orange3.git

cd orange3

# install requirements
pip install numpy
pip install scipy
pip install -r requirements-core.txt  # For Orange Python library
# pip install -r requirements-gui.txt   # For Orange GUI

# pip install -r requirements-sql.txt   # To use SQL support


# install orange
pip install -e .

# go back to add-on dir
cd $TRAVIS_BUILD_DIR
