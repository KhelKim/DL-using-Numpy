# !/usr/bin/env bash

mkdir ../data ../data/chap03/

kaggle datasets download -d uciml/faulty-steel-plates
unzip faulty-steel-plates.zip
mv faults.csv ../data/chap03/
rm faulty-steel-plates.zip
