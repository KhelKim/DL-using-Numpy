# !/usr/bin/env bash

mkdir ../data ../data/chap01/
kaggle datasets download -d maik3141/abalone
unzip abalone.zip
mv abalone.data.csv abalone.csv
mv abalone.csv ../data/chap01/
rm abalone.zip
