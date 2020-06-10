# !/usr/bin/env bash

mkdir ../data ../data/chap05/

kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers-recognition.zip
mv flowers ../data/chap05/
rm flowers-recognition.zip
