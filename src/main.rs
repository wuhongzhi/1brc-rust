use anyhow::{Ok, Result};
use clap::{Args, Parser};
use core::str;
use fork::{Fork, fork};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashSet,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write, stdout},
    os::fd::FromRawFd,
    process::exit,
    time::SystemTime,
};

use crate::r#gen::Mmap;

/// 1BRC for RUST
#[derive(Parser)]
#[command(version, about)]
enum Cli {
    /// Generate data from template file
    Gen(GenerateArg),

    /// Run benchmark
    Bench(BenchArg),
}
#[derive(Args)]
struct GenerateArg {
    /// Record size
    #[arg(short, long, default_value_t = 1_000_000)]
    size: u64,

    /// cities used for data file
    #[arg(short, long, default_value_t = 10_000)]
    cities: usize,

    /// template file
    #[arg(short, long, default_value = "./data/weather_stations.csv")]
    template: String,

    /// data file
    #[arg(short, long, default_value = "./data/measurements.txt")]
    data: String,

    /// write data line by line
    #[arg(short, long)]
    legency: bool,
}
#[derive(Args)]
struct BenchArg {
    /// data file
    #[arg(short, long, default_value = "./data/measurements.txt")]
    data: String,
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Gen(g) => generate(g),
        Cli::Bench(b) => bench(b),
    }
}
fn generate(argv: GenerateArg) -> Result<()> {
    let mut data = String::new();
    let data = {
        let mut file = File::open(argv.template)?;
        file.read_to_string(&mut data)?;
        data.as_bytes()
    };
    let cities: Vec<bench::City<'_>> = bench::decode_lines(data)
        .1
        .into_keys()
        .take(argv.cities)
        .collect();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&argv.data)?;
    let mut set = HashSet::new();
    let mut len = {
        macro_rules! r {
            ($e:expr) => {
                unsafe { libc::rand() } as usize % $e
            };
        }
        macro_rules! lines {
            ($e:expr) => {
                let bar = ProgressBar::new(argv.size);
                bar.set_message("Generate data ...");
                bar.set_style(
                    ProgressStyle::with_template(
                        "{msg} {wide_bar:.cyan/blue} {pos:>9}/{len:9} [{elapsed_precise}]",
                    )
                    .unwrap(),
                );
                for _ in 0..argv.size {
                    let city = &cities[r!(cities.len())];
                    let temp = format!(
                        ";{}{}.{}\n",
                        if r!(100) < 50 { "-" } else { "" },
                        r!(100),
                        r!(10),
                    );
                    $e.write_all(&city)?;
                    $e.write_all(temp.as_bytes())?;
                    bar.inc(1);
                    set.insert(city);
                }
            };
        }
        if argv.legency {
            let mut file = BufWriter::new(file);
            lines!(file);
            file.flush()?;
            file.seek(SeekFrom::End(0))? as f64
        } else {
            let mut file = Mmap::open::<true>(file)?;
            lines!(file);
            file.flush()?;
            file.finish() as f64
        }
    };
    let units = ["Bytes", "KiB", "MiB", "GiB", "TiB"];
    let mut i = 0;
    while len >= 1024_f64 && i < units.len() - 1 {
        len /= 1024_f64;
        i += 1;
    }
    len *= 100f64;
    eprintln!(
        "Data size: {:.2} {} with {} cites",
        len.round() / 100f64,
        units[i],
        set.len()
    );
    Ok(())
}

fn bench(argv: BenchArg) -> Result<()> {
    let clock = SystemTime::now();

    let mut pipe_fds = [0i32; 2];
    unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
    let mut buf = String::with_capacity(300_000);
    match fork()? {
        Fork::Parent(_child) => {
            unsafe { libc::close(pipe_fds[1]) }; // Close write end
            let mut reader = unsafe { std::fs::File::from_raw_fd(pipe_fds[0]) };
            reader.read_to_string(&mut buf)?;
            stdout().write_all(buf.as_bytes())?;
        }
        Fork::Child => {
            unsafe { libc::close(pipe_fds[0]) }; // Close read end
            let data = self::r#gen::Mmap::open::<false>(File::open(argv.data)?)?;
            let (records, result) = bench::reduce(&data)?;
            //format & detail report
            buf.push('{');
            for (id, (city, weather)) in result.iter().enumerate() {
                if id != 0 {
                    buf.push_str(", ");
                }
                buf.push_str(&format!("{city}={weather}"));
            }
            buf.push_str("}\n");
            buf.push_str(
                format!(
                    "Result in {:.3}ms with {} cities {records} records\n",
                    clock.elapsed()?.as_millis(),
                    result.len(),
                )
                .as_str(),
            );
            let mut writer = unsafe { File::from_raw_fd(pipe_fds[1]) };
            writer.write_all(buf.as_bytes())?;
        }
    }
    exit(0);
}

mod r#gen {
    use anyhow::{Result, bail};
    use std::{
        fs::File,
        io::{Error, Write},
        mem::ManuallyDrop,
        ops::Deref,
        os::fd::AsRawFd,
    };

    trait Remaining {
        fn remain(&self) -> usize;
    }

    #[derive(Default, Debug)]
    struct RawData {
        ptr: *mut u8,
        data: ManuallyDrop<Vec<u8>>,
    }

    impl RawData {
        fn new(ptr: *mut u8, length: usize, capacity: usize) -> Self {
            Self {
                ptr,
                data: ManuallyDrop::new(unsafe { Vec::from_raw_parts(ptr, length, capacity) }),
            }
        }
    }

    impl Remaining for RawData {
        fn remain(&self) -> usize {
            self.data.capacity() - self.data.len()
        }
    }

    impl Write for RawData {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.data.write(buf)?;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Deref for RawData {
        type Target = [u8];
        fn deref(&self) -> &Self::Target {
            &self.data
        }
    }
    impl Drop for RawData {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                munmap(self.ptr as _, self.data.capacity());
            }
        }
    }

    #[derive(Debug)]
    pub struct Mmap {
        file: File,
        chunk: usize,
        offset: u64,
        length: u64,
        write: bool,
        inner: RawData,
    }

    impl Mmap {
        pub fn open<const WRITE: bool>(file: File) -> Result<Self> {
            let length = file.metadata()?.len() as _;
            let write = WRITE;
            if write {
                let mut chunk = 64 * 1024usize;
                unsafe {
                    let page_size = libc::sysconf(libc::_SC_PAGESIZE);
                    if page_size == -1 {
                        bail!("Unknown page size");
                    }
                    //make user page align
                    chunk *= page_size as usize;
                }
                Ok(Self {
                    file,
                    chunk,
                    offset: 0,
                    length,
                    write,
                    inner: RawData::default(),
                })
            } else {
                let mut chunk = usize::MAX;
                if chunk as u64 > length {
                    chunk = length as usize
                }
                let offset = 0u64;
                let ptr = mmap::<false>(&file, chunk, offset)?;
                Ok(Self {
                    file,
                    chunk,
                    offset,
                    length,
                    write,
                    inner: RawData::new(ptr, chunk, chunk),
                })
            }
        }
    }
    impl Mmap {
        pub fn finish(&mut self) -> u64 {
            if self.write {
                self.length = self.offset + self.inner.len() as u64;
                self.file.set_len(self.length).unwrap();
            }
            self.length
        }
        fn next_map(&mut self) -> std::io::Result<()> {
            if !self.inner.ptr.is_null() {
                self.offset += self.chunk as u64;
            }
            self.length = self.offset + self.chunk as u64;
            self.file.set_len(self.length)?;
            let ptr = mmap::<true>(&self.file, self.chunk, self.offset)?;
            self.inner = RawData::new(ptr, 0, self.chunk);

            Ok(())
        }
    }
    fn munmap(ptr: *mut u8, capacity: usize) {
        unsafe {
            libc::munmap(ptr as _, capacity);
        }
    }
    fn mmap<const WRITE: bool>(file: &File, chunk: usize, offset: u64) -> std::io::Result<*mut u8> {
        unsafe {
            let mut ptr = std::ptr::null_mut();
            ptr = libc::mmap(
                ptr,
                chunk,
                libc::PROT_READ | if WRITE { libc::PROT_WRITE } else { 0 },
                libc::MAP_SHARED,
                file.as_raw_fd(),
                offset as _,
            );
            if ptr == libc::MAP_FAILED {
                return Err(Error::from_raw_os_error(*libc::__errno_location() as _));
            }
            Ok(ptr as _)
        }
    }

    impl Drop for Mmap {
        fn drop(&mut self) {
            self.finish();
        }
    }

    impl Deref for Mmap {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            &self.inner.data
        }
    }

    impl Write for Mmap {
        fn write(&mut self, mut bytes: &[u8]) -> std::io::Result<usize> {
            let len = bytes.len();
            while !bytes.is_empty() {
                let remain = self.inner.remain();
                if remain > bytes.len() {
                    self.inner.write(bytes)?;
                    break;
                } else if remain > 0 {
                    self.inner.write(&bytes[..remain])?;
                    bytes = &bytes[remain..];
                }
                self.next_map()?;
            }
            Ok(len)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
mod bench {
    use anyhow::{Ok, Result};
    use core::{slice, str};
    use rapidhash::{HashMapExt, RapidHashMap};
    use std::{
        collections::{BTreeMap, LinkedList},
        fmt::{self, Display, Formatter},
        hash::Hash,
        num::NonZeroUsize,
        ops::{AddAssign, Deref},
        sync::{Arc, Mutex, mpsc::channel},
        thread,
    };

    #[derive(Clone, Copy)]
    pub struct Weather {
        min: i64,
        max: i64,
        sum: i64,
        count: usize,
    }
    impl Weather {
        fn new(value: i64) -> Self {
            Self {
                min: value,
                max: value,
                sum: value,
                count: 1,
            }
        }
    }
    impl AddAssign for Weather {
        fn add_assign(&mut self, rhs: Self) {
            self.sum += rhs.sum;
            self.count += rhs.count;
            if self.max < rhs.max {
                self.max = rhs.max;
            }
            if self.min > rhs.min {
                self.min = rhs.min;
            }
        }
    }

    impl Display for Weather {
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            write!(
                f,
                "{:.1}/{:.1}/{:.1}",
                self.min as f64 / 10f64,
                (self.sum as f64 / self.count as f64).round() / 10f64,
                self.max as f64 / 10f64,
            )
        }
    }

    macro_rules! peek {
        ($e:expr) => {
            unsafe { *($e).as_ptr() }
        };
        ($e:expr, $i:expr) => {
            unsafe { *($e).as_ptr().add($i) }
        };
    }

    // #[repr(transparent)]
    #[derive(Eq, PartialOrd, Ord)]
    pub struct City<'a> {
        pub name: &'a [u8],
    }
    impl<'a> City<'a> {
        pub fn new(name: &'a [u8]) -> Self {
            Self { name }
        }
    }
    impl<'a> Hash for City<'a> {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            if self.name.len() < 8 {
                self.name.hash(state);
            } else {
                self.name[..8].hash(state);
            }
        }
    }
    impl<'a> PartialEq for City<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.name == other.name

            // use libc::memcmp;
            // self.name.len() == other.name.len()
            //     && unsafe {
            //         memcmp(
            //             self.name.as_ptr() as *const _,
            //             other.name.as_ptr() as *const _,
            //             self.name.len(),
            //         ) == 0
            //     }
        }
    }
    impl<'a> Display for City<'a> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.write_str(str::from_utf8(self.name).unwrap())
        }
    }

    impl<'a> Deref for City<'a> {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            self.name
        }
    }

    struct Scanner<'a> {
        data: Mutex<LinkedList<&'a [u8]>>,
        size: usize,
    }
    impl<'a> Scanner<'a> {
        fn new(data: &'a [u8], _parts: NonZeroUsize) -> Self {
            let mut v = vec![data];
            // let parts = (data.len() / (1 << 20)).ilog2();
            let parts = _parts.ilog2();
            for _i in 0..parts {
                let mut t = vec![];
                for data in v {
                    let end = data.len();
                    if end > 2 {
                        let mut i = end / 2;
                        while i < end && peek!(data, i) != b'\n' {
                            i += 1;
                        }
                        if i == end {
                            t.push(data);
                        } else {
                            t.push(&data[..=i]);
                            t.push(&data[i + 1..]);
                        }
                    } else {
                        t.push(data);
                    }
                }
                v = t;
            }
            let mut l = LinkedList::new();
            l.extend(v);
            Self {
                size: l.len(),
                data: Mutex::new(l),
            }
        }
        fn next(&self) -> Option<&'a [u8]> {
            let mut lock = self.data.lock().unwrap();
            lock.pop_front()
        }
        fn size(&self) -> usize {
            self.size
        }
    }

    type WeatherHashMap<'a> = RapidHashMap<City<'a>, Weather>;

    pub fn reduce(data: &[u8]) -> Result<(u64, BTreeMap<City<'_>, Weather>)> {
        let (tasks, rx) = {
            let jobs = num_cpus::get_physical();
            let scanner = Arc::new(Scanner::new(
                unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) },
                jobs.try_into().unwrap(),
            ));
            let jobs = jobs.min(scanner.size());
            // eprintln!("Parallel {jobs} jobs reducing {} parts", scanner.size());
            let (tx, rx) = channel::<Option<(u64, WeatherHashMap)>>();
            let mut tasks = vec![];
            for _ in 0..jobs {
                let sc1 = scanner.clone();
                let tx1 = tx.clone();
                tasks.push(thread::spawn(move || {
                    while let Some(part) = sc1.next() {
                        if !part.is_empty() {
                            tx1.send(Some(decode_lines(part))).unwrap();
                        }
                    }
                    tx1.send(None).unwrap();
                }));
            }
            (tasks, rx)
        };

        let mut records = 0;
        let mut finish = 0;
        let mut result = BTreeMap::new();
        while finish < tasks.len() {
            match rx.recv()? {
                Some((r, batch)) => {
                    records += r;
                    for (city, value) in batch {
                        result
                            .entry(city)
                            .and_modify(|v| *v += value)
                            .or_insert(value);
                    }
                }
                None => finish += 1,
            }
        }

        // just wait all, normal they all down here.
        tasks.into_iter().for_each(|t| t.join().unwrap());

        Ok((records, result))
    }

    #[inline(always)]
    fn find_chr(data: &[u8], chr: u8) -> Option<usize> {
        // let (mut i, end) = (0, data.len());
        // while i < end && peek!(data, i) != chr {
        //     i += 1;
        // }
        // if i == end { None } else { Some(i) }

        let p1 = data.as_ptr();
        let p2 = unsafe { libc::memchr(p1 as *const _, chr as _, data.len()) };
        if !p2.is_null() {
            Some(p2.addr() - p1.addr())
        } else {
            None
        }
    }
    pub fn decode_lines<'a>(mut data: &'a [u8]) -> (u64, WeatherHashMap<'a>) {
        let mut batch = WeatherHashMap::with_capacity(10_000);
        let mut records = 0;
        #[inline(always)]
        fn compute<'a>(batch: &mut WeatherHashMap<'a>, city: &'a [u8], temp: &'a [u8]) {
            batch
                .entry(City::new(city))
                .and_modify(|w| {
                    let value = parse_number(temp);
                    w.count += 1;
                    w.sum += value;
                    if value > w.max {
                        w.max = value
                    } else if value < w.min {
                        w.min = value;
                    }
                })
                .or_insert_with(|| Weather::new(parse_number(temp)));
        }
        const EMPTY: &[u8] = &[];
        while !data.is_empty() {
            let mut city = EMPTY;
            if peek!(data) != b'#'
                && let Some(p) = find_chr(data, b';')
            {
                city = &data[..p];
                data = &data[p + 1..];
            }
            match find_chr(data, b'\n') {
                Some(p) => {
                    if !city.is_empty() {
                        compute(&mut batch, city, &data[..p]);
                        records += 1;
                    }
                    data = &data[p + 1..];
                }
                None => break,
            }
        }
        (records, batch)
    }

    #[inline(always)]
    fn parse_number(mut temp: &[u8]) -> i64 {
        let x = match peek!(temp) {
            b'-' => {
                temp = &temp[1..];
                |v: i16| -v as i64
            }
            _ => |v: i16| v as i64,
        };

        macro_rules! m1 {
            ($e:expr) => {
                (10_i16 * $e)
            };
        }
        macro_rules! m2 {
            ($e:expr) => {
                (100_i16 * $e)
            };
        }
        macro_rules! d {
            ($e:expr) => {
                peek!($e) as i16 & 0x0f
            };
            ($e:expr, $i:expr) => {
                peek!($e, $i) as i16 & 0x0f
            };
        }
        if temp.len() > 4 {
            temp = &temp[..4]
        }
        x(match temp.len() {
            //1.1 => 11
            3 => m1!(d!(temp)) + d!(temp, 2),
            //11.1 => 111
            4 => m2!(d!(temp)) + m1!(d!(temp, 1)) + d!(temp, 3),
            x => unreachable!("bad number {x}"),
        })
    }

    #[cfg(test)]
    pub mod tests {
        use super::{City, decode_lines, parse_number};

        #[test]
        pub fn test_parse_number() {
            let val = parse_number("10.9".as_bytes());
            assert_eq!(val, 109);
            let val = parse_number("-10.9".as_bytes());
            assert_eq!(val, -109);
            let val = parse_number("1.0".as_bytes());
            assert_eq!(val, 10);
            let val = parse_number("-1.0".as_bytes());
            assert_eq!(val, -10);
        }

        #[test]
        pub fn test_decode() {
            let data = "aaaaaaaa;-10.0\naaaaaaaa;26.0\ndef;2.1\n".as_bytes();
            let (_, m) = decode_lines(data);
            assert_eq!(m.len(), 2);
            let r = m.get(&City::new("aaaaaaaa".as_bytes())).unwrap();
            assert_eq!(r.count, 2);
            assert_eq!(r.min, -100);
            assert_eq!(r.max, 260);
            assert_eq!(r.sum, 160);
        }
    }
}
