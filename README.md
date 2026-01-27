# The One Billion Row Challenge for RUST

https://github.com/gunnarmorling/1brc

```
Usage: rs_1brc <COMMAND>

Commands:
  gen    Generate data from template file
  bench  Run benchmark
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Generate data from template file

```
Usage: rs_1brc gen [OPTIONS]

Options:
  -s, --size <SIZE>          Record size [default: 1000000000]
  -c, --cities <CITIES>      cities used for data file, max 10,000 cities [default: 413]
      --template <TEMPLATE>  template file [default: ./data/weather_stations.csv]
      --data <DATA>          data file [default: ./data/measurements.txt]
  -l, --legacy               write data line by line
      --hugepages            data file in hugetlbfs
  -h, --help                 Print help
```

## Run benchmark

```
Usage: rs_1brc bench [OPTIONS]

Options:
      --data <DATA>        data file [default: ./data/measurements.txt]
  -s, --slice <SLICE>      slice size, default to file size / workers
  -w, --workers <WORKERS>  parallel workers, default to cpu cores
  -d, --dry-run            dry-run without map/reduce
  -m, --mode <MODE>        mode (0: simd-scan, 1: simd-batch, 2: simd-sequence) [default: 2]
      --hugepages          data file in hugetlbfs
  -h, --help               Print help
```
