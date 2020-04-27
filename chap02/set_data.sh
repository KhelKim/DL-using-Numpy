# !/usr/bin/env bash

mkdir ../data ../data/chap02/

kaggle datasets download -d pavanraj159/predicting-a-pulsar-star
unzip predicting-a-pulsar-star.zip
mv pulsar_stars.csv ../data/chap02/
rm predicting-a-pulsar-star.zip
