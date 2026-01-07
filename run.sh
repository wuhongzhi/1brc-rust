#!/bin/bash
export RUSTFLAGS="-Ctarget-cpu=native"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmimalloc.so.3
cd $(dirname $0)
if [ "$1" == "g" ]; then
    shift
    cargo flamegraph --release -- bench $@> /dev/null
elif [ "$1" == 'd' ]; then    
    shift
    cargo run --release -- gen $@
else
    target=""
    if [ "$1" == 'x' ]; then
        shift
        target="x86_64-unknown-linux-musl"
    fi
    app="./target/${target}/release/rs_1brc"
    if [ "$1" == 'b' ]; then
        shift
        for s in {1..11}; do
            time $app bench $@ > /dev/null
            sleep 1
        done |& grep real | sed -e 's/real//' | sort | nl
    elif [ "$1" == 'p' ]; then
        shift
        perf stat $app bench $@ > /dev/null
    else
        if [ -z "$target"]; then
            time cargo run --release -- bench $@
        else
            time cargo run --release --target $target -- bench $@
        fi
    fi
fi    