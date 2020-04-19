# !/usr/bin/env bash

# sh make_structure.sh [만들폴더 이름]
mkdir $1
mkdir $1/codes
touch $1/README.md $1/set_data.sh
touch $1/codes/implement.py $1/codes/run.py
