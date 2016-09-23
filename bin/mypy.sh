#!/bin/bash

bin_dir=$(dirname $0)

export PYTHONPATH=$(readlink -f $(dirname $bin_dir)):$PYTHONPATH

python "$@"
#pydnn.sh "$@"
