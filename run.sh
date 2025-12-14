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
    done |& grep real | sed 's/real//' | cut -f2 -d'm' | cut -f1 -d's' | sort | head -n9 | tail -n8 | \
        awk '{count+=$1} END{print count/NR}'
else
    cargo run --release -- bench
fi
