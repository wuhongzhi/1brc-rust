#!/bin/bash
cd $(dirname $0)
if [ "$1" == "g" ]; then
    cargo flamegraph --release -- bench > /dev/null
elif [ "$1" == 'd' ]; then    
    shift
    cargo run --release -- gen $@
elif [ "$1" == 'b' ]; then
    shift
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmimalloc.so.3
    for s in {1..5}; do
       time ./target/release/rs_1brc bench $@ > /dev/null
    done |& grep real | sed -e 's/real//' -e 's/s$//' -e 's/m/\*60+/' | \
        bc | sort | head -n4 | tail -n3 | \
        awk '{count+=$1} END{print count/NR}'
else
    time cargo run --release -- bench $@
fi
