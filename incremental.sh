#!/bin/bash

model=$1
formula=$2

function runStorm () {
    local file=$1
    storm --prism $file --prop "$formula" | \
                         grep 'Result (for initial states)' | \
                         cut -d ':' -f 2
}

out_file=$(basename `echo $model | cut -d '.' -f 1`.dat)

echo '# n actual freq dirichlet' &> $out_file

for file in `ls out/dirichlet_model*.prism | sort`; do
    python main.py -N 3000 --batches 15 $model &> /dev/null
    actual=`runStorm out/model.prism`
    freq=`runStorm out/frequentist_model.prism`
    dirichlet=`runStorm out/dirichlet_model_0.prism`
    echo $n $actual $freq $dirichlet &>> $out_file
done
