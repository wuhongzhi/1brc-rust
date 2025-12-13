#!/bin/bash
cd $(dirname $0)
if [ "$1" == "g" ]; then
    cargo flamegraph --release -- bench > /dev/null
elif [ "$1" == 'd' ]; then    
    shift
    cargo run --release -- gen $@
elif [ "$1" == 'b' ]; then
    for s in {1..10}; do
       time ./target/release/rs_1brc bench > /dev/null
    done
else
    cargo run --release -- bench > /dev/null
fi
