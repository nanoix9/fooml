#!/bin/bash

bin_dir=$(dirname $0)

export PATHONPATH=$PATHONPATH:$(readlink -f $(dirname $bin_dir))
echo $PATHONPATH

python "$@"
#pydnn.sh "$@"
