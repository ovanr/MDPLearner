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

for n in 50 100 200 500 1000 2000 4000 6000 8000; do
    rm out/*
    python main.py -N $n --batches 1 --laplace-smoothing 0.5 --init-alpha 2.5 $model 
    actual=`runStorm out/model.prism`
    freq=`runStorm out/frequentist_model.prism`
    dirichlet=`runStorm out/dirichlet_model_*.prism`
    echo $n $actual $freq $dirichlet &>> $out_file
done
