#!/bin/bash
# set -x
export RUSTFLAGS="-Ctarget-cpu=x86-64-v3"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmimalloc.so.3
cd $(dirname $0)
if [ "$1" == "g" ]; then
    shift
    cargo flamegraph \
        --skip-after "rs_1brc::bench::decode_lines_a" \
        --skip-after "rs_1brc::bench::decode_lines_b" \
        --skip-after "<rs_1brc::bench::City as core::cmp::PartialEq>::eq" \
        --skip-after "<rapidhash::inner::state::random_state::RandomState<false, true, false, false> as core::hash::BuildHasher>::hash_one::<rs_1brc::bench::City>" \
        --release -- bench $@> /dev/null
elif [ "$1" == 'd' ]; then
    shift
    cargo run --release -- gen $@
else
    target=""
    if [ "$1" == 'x' ]; then
        shift
        target="x86_64-unknown-linux-musl/"
    fi
    app="./target/${target}release/rs_1brc"
    if [ "$1" == 'b' ]; then
        shift
        for s in {1..11}; do
            time $app bench $@ > /dev/null
            sleep 0.5s
        done |& grep real | sed -e 's/real//' | sort | nl
    elif [ "$1" == 'p' ]; then
        shift
        perf stat $app bench $@ > /dev/null
    else
        if [ -n "$target" ]; then
            target="--target $(echo $target | sed 's/\///')"
        fi
        time cargo run --release $target -- bench $@
    fi
fi

