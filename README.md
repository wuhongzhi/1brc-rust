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
  -s, --size <SIZE>          Record size [default: 1000000]
  -c, --cities <CITIES>      cities used for data file [default: 10000]
  -t, --template <TEMPLATE>  template file [default: ./data/weather_stations.csv]
  -d, --data <DATA>          data file [default: ./data/measurements.txt]
  -l, --legency              write data line by line
  -h, --help                 Print help
```

## Run benchmark

```
Usage: rs_1brc bench [OPTIONS]

Options:
  -d, --data <DATA>  data file [default: ./data/measurements.txt]
  -h, --help         Print help
```
