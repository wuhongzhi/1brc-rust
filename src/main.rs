#![feature(portable_simd)]
#![feature(test)]
use crate::{bench::ThousandSep, r#gen::Mmap};
use anyhow::Result;
use clap::{Args, Parser};
use core::str;
use fork::{Fork, fork};
use indicatif::{ProgressBar, ProgressStyle};
use std::{
    collections::HashSet,
    env::args,
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write, stdout},
    mem::ManuallyDrop,
    os::fd::FromRawFd,
    time::SystemTime,
};

macro_rules! ptr_add {
    ($e: expr, $i: expr) => {
        ($e as usize as isize + ($i)  as isize) as usize as *const u8
    };
}
macro_rules! ptr_inc {
    ($e: expr, $i: expr) => {
        $e = ($e as usize as isize + ($i)  as isize) as usize as *const u8
    };
}
macro_rules! pre_slice {
    ($d:expr, $i:expr) => {
        unsafe { slice::from_raw_parts($d.as_ptr(), $i) }
    };
}
macro_rules! pst_slice {
    ($d:expr, $i:expr) => {
        unsafe {
            let off = $i;
            let len = $d.len() - off;
            slice::from_raw_parts(ptr_add!($d.as_ptr(), off), len)
        }
    };
}
macro_rules! mid_slice {
    ($d:expr, $s:expr, $e:expr) => {
        unsafe { slice::from_raw_parts(ptr_add!($d.as_ptr(), $s), $e - ($s)) }
    };
}
const HASH_SIZE: usize = 1 << 15;
const DEFULAT_CITIES: usize = 413;

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
    #[arg(short, long, default_value_t = 1_000_000_000)]
    size: u64,

    /// cities used for data file, max 10,000 cities
    #[arg(short, long, default_value_t = DEFULAT_CITIES, value_parser = max_cities)]
    cities: usize,

    /// template file
    #[arg(long, default_value = "./data/weather_stations.csv")]
    template: String,

    /// data file
    #[arg(long, default_value = "./data/measurements.txt")]
    data: String,

    /// write data line by line
    #[arg(short, long)]
    legacy: bool,

    /// data file in hugetlbfs
    #[arg(long)]
    hugepages: bool,
}
fn max_cities(s: &str) -> Result<usize> {
    Ok(s.parse::<usize>()?.clamp(1, 10_000))
}
#[derive(Args)]
struct BenchArg {
    /// data file
    #[arg(long, default_value = "./data/measurements.txt")]
    data: String,

    /// slice size, default to file size / workers
    #[arg(short, long)]
    slice: Option<usize>,

    /// parallel workers, default to cpu cores
    #[arg(short, long)]
    workers: Option<usize>,

    /// dry-run without map/reduce
    #[arg(short, long)]
    dry_run: bool,

    /// mode (0: simd-scan, 1: simd-batch, 2: simd-sequence)
    #[arg(short, long, default_value_t = 2)]
    mode: usize,

    /// data file in hugetlbfs
    #[arg(long)]
    hugepages: bool,
}

pub fn main() -> Result<()> {
    //bypass all clap phase
    if args().any(|x| x == "x") {
        bench(BenchArg {
            data: "./data/measurements.txt".into(),
            slice: None,
            workers: None,
            dry_run: false,
            hugepages: false,
            mode: 2,
        })
    } else {
        match Cli::parse() {
            Cli::Gen(g) => generate(g),
            Cli::Bench(b) => bench(b),
        }
    }
}
fn generate(argv: GenerateArg) -> Result<()> {
    let data: String = {
        if argv.cities == DEFULAT_CITIES {
            include_str!("../data/default_weather_stations.csv").into()
        } else {
            let mut data = String::new();
            let mut file = File::open(argv.template)?;
            file.read_to_string(&mut data)?;
            data
        }
    };
    macro_rules! rand {
        ($e:expr) => {
            unsafe { libc::rand() } as usize % $e
        };
    }
    let mut cities: Vec<(&str, bool)> = {
        let mut cities = HashSet::new();
        data.lines().for_each(|line| {
            if !line.starts_with("#")
                && let Some(city) = line.split(";").next()
            {
                cities.insert(city);
            }
        });
        let skip = if cities.len() > argv.cities {
            rand!(cities.len() - argv.cities) as usize
        } else {
            0
        };
        cities
            .into_iter()
            .skip(skip)
            .take(argv.cities)
            .map(|city| (city, false))
            .collect()
    };
    let mut len = {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&argv.data)?;
        macro_rules! lines {
            ($e:expr) => {
                let bar = ProgressBar::new(argv.size);
                bar.set_message("Generate data ...");
                bar.set_style(
                    ProgressStyle::with_template(
                        "{msg} {wide_bar:.green/blue} {pos:>9}/{len:9} [{elapsed_precise}]",
                    )
                    .unwrap(),
                );
                let len = cities.len();
                for j in 0..argv.size {
                    let city = &mut cities[rand!(len)];
                    let temp = format!(
                        ";{}{:.1}\n",
                        if rand!(100) < 50 { "-" } else { "" },
                        rand!(1000) as f32 / 10f32,
                    );
                    $e.write_all(city.0.as_bytes())?;
                    $e.write_all(temp.as_bytes())?;
                    if j % 10_000 == 0 && j != 0 {
                        bar.inc(10_000);
                    }
                    city.1 = true
                }
            };
        }
        if argv.legacy {
            let mut file = BufWriter::new(file);
            lines!(file);
            file.flush()?;
            file.seek(SeekFrom::End(0))? as f64
        } else {
            let mut file = Mmap::open::<true>(file, argv.hugepages)?;
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

    eprintln!(
        "Final size {} {} with {} cities",
        ((len * 100f64).round() / 100f64).format(3),
        units[i],
        cities.iter().filter(|f| f.1).count().format(0)
    );
    Ok(())
}

fn bench(argv: BenchArg) -> Result<()> {
    fn new_buf() -> ManuallyDrop<Vec<u8>> {
        ManuallyDrop::new(Vec::with_capacity(512_000))
    }
    let clock = SystemTime::now();
    let mut pipe_fds = [0i32; 2];
    unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
    match fork()? {
        Fork::Parent(child) => unsafe {
            libc::setpriority(libc::PRIO_PROCESS, child as u32, libc::PRIO_MIN);
            libc::fcntl(pipe_fds[0], libc::F_SETPIPE_SZ, 1 << 20);
            libc::close(pipe_fds[1]);
            let mut buf = new_buf();
            File::from_raw_fd(pipe_fds[0]).read_to_end(&mut buf)?;
            stdout().lock().write_all(&buf)?;
            libc::exit(0);
        },
        Fork::Child => {
            match Mmap::open::<false>(File::open(argv.data)?, argv.hugepages) {
                Ok(data) => {
                    bench::reduce(&data, argv.slice, argv.workers, argv.dry_run, argv.mode)?
                        .write(unsafe { File::from_raw_fd(pipe_fds[1]) }, new_buf())?
                        .wait(clock)?;
                }
                Err(e) => {
                    eprintln!("{e:?}")
                }
            }
            Ok(())
        }
    }
}

mod r#gen {
    use anyhow::{Result, bail};
    use std::{
        fs::File,
        io::{Error, Write},
        mem::ManuallyDrop,
        ops::Deref,
        os::fd::AsRawFd,
        slice,
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
        hugepages: bool,
    }

    impl Mmap {
        pub fn open<const WRITE: bool>(file: File, hugepages: bool) -> Result<Self> {
            let length = file.metadata()?.len() as _;
            let write = WRITE;
            if write {
                let mut chunk = 16 * 1024;
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
                    hugepages,
                    inner: RawData::default(),
                })
            } else {
                if length == 0 {
                    bail!("Empty file");
                }
                let mut chunk = usize::MAX;
                if chunk as u64 > length {
                    chunk = length as usize
                }
                let offset = 0u64;
                let ptr = mmap::<false>(&file, chunk, offset, hugepages)?;
                let length = if hugepages {
                    unsafe { *ptr.add(chunk - 8).cast::<u64>() }.to_le()
                } else {
                    length
                };
                Ok(Self {
                    file,
                    chunk,
                    offset,
                    length,
                    write,
                    hugepages,
                    inner: RawData::new(ptr, length as usize, chunk),
                })
            }
        }
    }
    impl Mmap {
        pub fn finish(&mut self) -> u64 {
            if self.write {
                self.write = false;
                let length = self.offset + self.inner.len() as u64;
                if self.hugepages {
                    if self.inner.remain() < 8 {
                        self.next_map().unwrap();
                    }
                    let pos = self.inner.data.capacity() - 8;
                    unsafe {
                        *self.inner.ptr.add(pos).cast::<u64>() = length.to_le();
                    }
                } else {
                    self.file.set_len(length).unwrap();
                    self.length = length;
                }
            }
            self.length
        }
        fn next_map(&mut self) -> std::io::Result<()> {
            if !self.inner.ptr.is_null() {
                self.offset += self.chunk as u64;
            }
            self.length = self.offset + self.chunk as u64;
            self.file.set_len(self.length)?;
            let ptr = mmap::<true>(&self.file, self.chunk, self.offset, self.hugepages)?;
            self.inner = RawData::new(ptr, 0, self.chunk);

            Ok(())
        }
    }
    fn munmap(ptr: *mut u8, capacity: usize) {
        unsafe {
            libc::munmap(ptr as _, capacity);
        }
    }
    fn mmap<const WRITE: bool>(
        file: &File,
        chunk: usize,
        offset: u64,
        hugepages: bool,
    ) -> std::io::Result<*mut u8> {
        unsafe {
            let mut ptr = std::ptr::null_mut();
            ptr = libc::mmap(
                ptr,
                chunk,
                libc::PROT_READ | if WRITE { libc::PROT_WRITE } else { 0 },
                libc::MAP_SHARED | if hugepages { libc::MAP_HUGETLB } else { 0 },
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
                    self.inner.write(pre_slice!(bytes, remain))?;
                    bytes = pst_slice!(bytes, remain);
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
    use crate::{DEFULAT_CITIES, HASH_SIZE};
    use anyhow::{Ok, Result};
    use core::slice;
    use proc_cpuinfo::CpuInfo;
    use rayon::prelude::*;
    use std::{
        cell::Cell, collections::BTreeMap, fmt::Display, fs::File, io::Write, mem::ManuallyDrop,
        ops::AddAssign, ptr::null, thread, time::SystemTime,
    };

    macro_rules! read {
        ($p:expr) => {
            unsafe { *$p }
        };
    }
    macro_rules! read_unaligned {
        ($p:expr, $ty:ty) => {
            unsafe { $p.cast::<$ty>().read_unaligned() }
        };
        ($p:expr, $i:expr, $ty:ty) => {
            unsafe { ptr_add!($p, $i).cast::<$ty>().read_unaligned() }
        };
    }

    type Fingerprint = (u32, u64);

    #[derive(PartialOrd, Ord, Eq)]
    pub struct City<'a> {
        cmp: Fingerprint,
        name: &'a [u8],
    }
    impl<'a> City<'a> {
        pub fn write(&self, buf: &mut Vec<u8>) {
            buf.extend_from_slice(self.name);
        }
        #[inline(always)]
        fn new(name: &'a [u8]) -> Self {
            Self::new_ex(name, u64::from_le(read_unaligned!(name.as_ptr(), u64)))
        }
        #[inline(always)]
        fn new_ex(name: &'a [u8], mut word1: u64) -> Self {
            let len = name.len();
            Self {
                name,
                cmp: (
                    len as u32,
                    if len <= 8 {
                        word1 <<= (8 - len) << 3;
                        word1 ^ word1.swap_bytes()
                    } else {
                        word1 ^ u64::from_le(read_unaligned!(name.as_ptr(), len - 8, u64))
                    },
                ),
            }
        }
        #[inline(always)]
        fn hash(&self) -> u64 {
            self.cmp.1
        }
    }
    impl<'a> From<&'a [u8]> for City<'a> {
        fn from(value: &'a [u8]) -> Self {
            Self::new(value)
        }
    }
    impl<'a> PartialEq for City<'a> {
        #[inline(always)]
        fn eq(&self, other: &Self) -> bool {
            self.cmp == other.cmp
        }
    }

    type Temperature = i32;
    #[derive(Clone, Copy)]
    pub struct Weather {
        cnt: u32,
        sum: i32,
        min: Temperature,
        max: Temperature,
        cmp: Fingerprint,
    }
    impl Weather {
        fn new(value: (Fingerprint, Temperature)) -> Self {
            Self {
                cmp: value.0,
                min: value.1,
                max: value.1,
                sum: value.1,
                cnt: 1,
            }
        }
        fn write(&self, buf: &mut Vec<u8>) {
            let mut avg = (self.sum as f64 / self.cnt as f64).round();
            if avg.abs() < 1.0 {
                avg = 0f64;
            }
            buf.extend_from_slice((self.min as f64 / 10f64).format(1).as_bytes());
            buf.push(b'/');
            buf.extend_from_slice((avg / 10f64).format(1).as_bytes());
            buf.push(b'/');
            buf.extend_from_slice((self.max as f64 / 10f64).format(1).as_bytes());
        }
    }
    impl From<(Fingerprint, Temperature)> for Weather {
        fn from(value: (Fingerprint, Temperature)) -> Self {
            Self::new(value)
        }
    }
    macro_rules! max_val {
        ($self: expr, $e: expr) => {{
            let this = &mut $self.max;
            *this = [*this, $e][(*this < $e) as usize];
        }};
    }
    macro_rules! min_val {
        ($self: expr, $e: expr) => {{
            let this = &mut $self.min;
            *this = [*this, $e][(*this > $e) as usize];
        }};
    }
    impl AddAssign for Weather {
        #[inline(always)]
        fn add_assign(&mut self, other: Self) {
            self.sum += other.sum;
            self.cnt += other.cnt;
            max_val!(self, other.max);
            min_val!(self, other.min);
        }
    }
    impl AddAssign<Temperature> for Weather {
        #[inline(always)]
        fn add_assign(&mut self, value: Temperature) {
            self.sum += value;
            self.cnt += 1;
            max_val!(self, value);
            min_val!(self, value);
        }
    }

    pub trait ThousandSep {
        fn format(&self, decimal: usize) -> String;
    }
    impl<T: Display> ThousandSep for T {
        fn format(&self, decimal: usize) -> String {
            let str = self.to_string();
            let mut v = str.split(".");
            let mut r = String::new();
            if let Some(mut v) = v.next() {
                let len = v.len() % 3;
                if len > 0 {
                    r.push_str(&v[..len]);
                    v = &v[len..];
                }
                while v.len() >= 3 {
                    if !r.is_empty() {
                        r.push(',');
                    }
                    r.push_str(&v[..3]);
                    v = &v[3..];
                }
            }
            if decimal > 0 {
                r.push('.');
                (match v.next() {
                    Some(v) => {
                        let len = decimal.min(v.len());
                        r.push_str(&v[..len]);
                        len
                    }
                    _ => 0,
                }..decimal)
                    .for_each(|_| r.push('0'));
            }
            r
        }
    }

    struct Scanner<'a> {
        data: &'a [u8],
        off: usize,
        slice: usize,
    }
    impl<'a> Scanner<'a> {
        fn new(data: &'a [u8], slice: usize) -> Self {
            Self {
                data,
                slice,
                off: 0,
            }
        }
    }
    impl<'a> Iterator for Scanner<'a> {
        type Item = &'a [u8];
        fn next(&mut self) -> Option<Self::Item> {
            let self_end = self.data.len();
            let off = self.off;
            if off < self_end {
                let expect_end = off + self.slice;
                let range = off..if expect_end < self_end {
                    let ptr = self.data.as_ptr();
                    let end = find_newline(ptr_add!(ptr, expect_end), ptr_add!(ptr, self_end));
                    if !end.is_null() {
                        end as usize - ptr as usize + 1
                    } else {
                        self_end
                    }
                } else {
                    self_end
                };
                self.off = range.end;
                Some(mid_slice!(self.data, range.start, range.end))
            } else {
                None
            }
        }
    }

    pub type WeatherMap<'a> = MyWeatherMap<'a>;

    pub struct MyWeatherMap<'a> {
        index: [u16; HASH_SIZE],
        weather: Vec<Weather>,
        city: Vec<City<'a>>,
    }

    impl<'a> Default for MyWeatherMap<'a> {
        fn default() -> Self {
            MyWeatherMap {
                index: [u16::MAX; HASH_SIZE],
                weather: Vec::with_capacity(DEFULAT_CITIES),
                city: Vec::with_capacity(DEFULAT_CITIES),
            }
        }
    }

    thread_local! {
        static HIS_MISS: Cell<usize> = const { Cell::new(0) };
    }
    #[allow(dead_code)]
    impl<'a> MyWeatherMap<'a> {
        pub fn len(&self) -> usize {
            self.city.len()
        }
        pub fn reset(&mut self) -> &mut Self {
            *self = Default::default();
            self
        }
        pub fn get(&self, city: &City<'static>) -> Option<Weather> {
            for (i, key) in self.city.iter().enumerate() {
                if key.eq(city) {
                    return Some(self.weather[i]);
                }
            }
            None
        }

        // TODO: refine this method
        #[inline(always)]
        pub fn put(&mut self, (city, value): (City<'a>, Temperature)) {
            const SIZE_MASK: usize = HASH_SIZE - 1;
            #[inline(always)]
            fn hash2index(index: u64) -> usize {
                let half = (index ^ (index >> 30)) as u32;
                (half ^ (half >> 15)) as usize & SIZE_MASK
            }
            let mut miss = 0;
            let mut slot = hash2index(city.hash() >> 4);
            loop {
                return match self.index[slot] {
                    u16::MAX => {
                        self.index[slot] = self.weather.len() as u16;
                        self.weather.push((city.cmp, value).into());
                        self.city.push(city);
                    }
                    index => {
                        let hit = &mut self.weather[index as usize];
                        if city.cmp == hit.cmp {
                            *hit += value;
                            #[cfg(feature = "hit_miss")]
                            HIS_MISS.with(|x| x.set(x.get() + miss));
                        } else if miss < SIZE_MASK {
                            miss += 1;
                            slot = (slot + 31) & SIZE_MASK;
                            continue;
                        } else {
                            panic!("Map is full!");
                        }
                    }
                };
            }
        }
    }

    impl<'a> From<MyWeatherMap<'a>> for BTreeMap<City<'a>, Weather> {
        fn from(val: MyWeatherMap<'a>) -> Self {
            let mut r = BTreeMap::default();
            for (i, city) in val.city.into_iter().enumerate() {
                r.insert(city, val.weather[i]);
            }
            r
        }
    }

    #[repr(transparent)]
    pub struct Reduce<'a>(((BTreeMap<City<'a>, Weather>, u64, usize), usize));

    impl<'a> Reduce<'a> {
        pub fn write(self, mut file: File, mut buf: ManuallyDrop<Vec<u8>>) -> Result<Self> {
            buf.push(b'{');
            self.0
                .0
                .0
                .iter()
                .enumerate()
                .for_each(|(id, (city, weather))| {
                    buf.extend_from_slice([", ", ""][(id == 0) as usize].as_bytes());
                    city.write(&mut buf);
                    buf.push(b'=');
                    weather.write(&mut buf);
                });
            buf.extend_from_slice("}\n".as_bytes());
            file.write_all(&buf)?;
            Ok(self)
        }
        pub fn wait(self, clock: SystemTime) -> Result<()> {
            let taken = clock.elapsed()?;
            #[cfg(feature = "hit_miss")]
            eprintln!(
                "Result in {}ms with {} lines and {} cities, on average of {:.3}ns/line, hit-miss {} miss-ratio {:.3}%",
                (taken.as_micros() as f64 / 1_000_f64).format(3),
                self.0.0.1.format(0),
                self.0.0.0.len().format(0),
                taken.as_nanos() as f64 / self.0.0.1.max(1) as f64 * self.0.1 as f64,
                self.0.0.2.format(0),
                self.0.0.2 as f64 * 100_f64 / self.0.0.1.max(1) as f64
            );
            #[cfg(not(feature = "hit_miss"))]
            eprintln!(
                "Result in {}ms with {} lines and {} cities, on average of {:.3}ns/line",
                (taken.as_micros() as f64 / 1_000_f64).format(3),
                self.0.0.1.format(0),
                self.0.0.0.len().format(0),
                taken.as_nanos() as f64 / self.0.0.1.max(1) as f64 * self.0.1 as f64,
            );
            Ok(())
        }
    }

    pub fn reduce(
        data: &[u8],
        slice: Option<usize>,
        workers: Option<usize>,
        dry_run: bool,
        mode: usize,
    ) -> Result<Reduce<'_>> {
        fn map<'a>(
            dry_run: bool,
            mode: usize,
        ) -> impl Fn(&'a [u8]) -> (BTreeMap<City<'a>, Weather>, u64, usize) {
            move |part| {
                let clock = SystemTime::now();
                let mut cities = WeatherMap::default();
                let total = match mode {
                    0 => decode_lines_a(part, &mut cities, dry_run),
                    1 => decode_lines_b(part, &mut cities, dry_run),
                    2 => decode_lines_c(part, &mut cities, dry_run),
                    _ => 0,
                };
                let miss = HIS_MISS.with(|x| x.take());
                if dry_run {
                    eprintln!(
                        "{:?} -> decode {} lines within {}ms, mode: {mode}",
                        thread::current().id(),
                        total.format(0),
                        (clock.elapsed().unwrap().as_micros() as f64 / 1_000_f64).format(3)
                    );
                }
                (cities.into(), total, miss)
            }
        }
        fn reduce<'a>(
            (mut result, c1, m1): (BTreeMap<City<'a>, Weather>, u64, usize),
            (batch, c2, m2): (BTreeMap<City<'a>, Weather>, u64, usize),
        ) -> (BTreeMap<City<'a>, Weather>, u64, usize) {
            batch.into_iter().for_each(|(city, value)| {
                result
                    .entry(city)
                    .and_modify(|weather| *weather += value)
                    .or_insert_with(|| value);
            });
            (result, c1 + c2, m1 + m2)
        }

        let (cpu_cores, cache_size) = {
            let proc = CpuInfo::read()?;
            match proc.cpus().last() {
                Some(cpu) => (
                    workers.unwrap_or(cpu.processor().unwrap_or(0) + 1).max(1),
                    cpu.cache_size().unwrap(),
                ),
                None => (1, 1 << 20),
            }
        };
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("decode-worker-{i}"))
            .num_threads(cpu_cores)
            .use_current_thread()
            .build_global()?;
        Ok(Reduce((
            Scanner::new(
                unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) },
                slice
                    .unwrap_or(data.len() / cpu_cores)
                    .max(cache_size / cpu_cores),
            )
            .par_bridge()
            .into_par_iter()
            .map(map(dry_run, mode))
            .reduce_with(reduce)
            .unwrap(),
            cpu_cores,
        )))
    }

    const CHR_NL: u8 = b'\n';
    const CHR_CM: u8 = b';';
    type FindBase = u64;
    type FindSimd = std::simd::u64x2;
    //min 7 byets one solt (ab;y.z\n)
    const SIMD_SOLTS: usize = (size_of::<FindSimd>() as f32 / 7f32).ceil() as usize;

    const BASE_SIZE: usize = size_of::<FindBase>();
    const BASE_MASK_CM: FindBase = FindBase::from_ne_bytes([CHR_CM; BASE_SIZE]);
    const BASE_MASK_NL: FindBase = FindBase::from_ne_bytes([CHR_NL; BASE_SIZE]);
    const BASE_MASK1: FindBase = FindBase::from_ne_bytes([0x01; BASE_SIZE]);
    const BASE_MASK2: FindBase = FindBase::from_ne_bytes([0x80; BASE_SIZE]);

    const FIND_SIZE: usize = size_of::<FindSimd>();
    const FIND_MASK_CM: FindSimd = FindSimd::splat(BASE_MASK_CM);
    const FIND_MASK_NL: FindSimd = FindSimd::splat(BASE_MASK_NL);
    const FIND_MASK1: FindSimd = FindSimd::splat(BASE_MASK1);
    const FIND_MASK2: FindSimd = FindSimd::splat(BASE_MASK2);

    struct Tokenizer<'a> {
        cache: [*const u8; SIMD_SOLTS],
        cache_offset: usize,
        cache_length: usize,

        data_ptr: *const u8,
        data_end: *const u8,
        data_align: *const u8,

        mask: &'a FindSimd,
        chr: u8,
    }
    impl<'a> Tokenizer<'a> {
        fn new(data: &'a [u8], chr: u8, mask: &'a FindSimd) -> Self {
            let data_ptr = data.as_ptr();
            Self {
                cache: [null(); SIMD_SOLTS],
                cache_offset: 0,
                cache_length: 0,
                data_end: ptr_add!(data_ptr, data.len()),
                data_align: data_ptr,
                data_ptr,
                mask,
                chr,
            }
            .align()
        }
        fn align(mut self) -> Self {
            const MASK: usize = FIND_SIZE - 1;
            let data_ptr = self.data_ptr;
            let data_end = self.data_end;
            if data_ptr < data_end {
                self.data_align = (data_end.addr() & !MASK) as *const u8;
                let unaligned = data_ptr.addr() & MASK;
                if unaligned != 0 {
                    self.fill_unaligned(unsafe {
                        ptr_add!(
                            data_ptr,
                            data_end
                                .offset_from_unsigned(data_ptr)
                                .min(FIND_SIZE - unaligned)
                        )
                    });
                };
            }
            self
        }
        fn fill_unaligned(&mut self, unaligned: *const u8) -> bool {
            let mut cache_length = 0;
            let mut data_ptr = self.data_ptr;
            let cache = &mut self.cache;
            let chr = self.chr;
            while data_ptr < unaligned {
                if read!(data_ptr) == chr {
                    cache[cache_length] = data_ptr;
                    cache_length += 1;
                }
                ptr_inc!(data_ptr, 1);
            }
            self.cache_length = cache_length;
            self.data_ptr = data_ptr;
            cache_length != 0
        }
        #[inline(always)]
        fn fill(&mut self) -> bool {
            let mut cache_length = 0;
            let mut data_ptr = self.data_ptr;
            let cache = &mut self.cache;
            let data_align = self.data_align;
            let mask = self.mask;
            while (cache_length == 0) & (data_ptr < data_align) {
                let value = mask ^ read!(data_ptr.cast::<FindSimd>());
                for mut x in ((value - FIND_MASK1) & !value & FIND_MASK2).to_array() {
                    while x != 0 {
                        let v = x.trailing_zeros();
                        x ^= 1 << v;
                        cache[cache_length] = ptr_add!(data_ptr, (v >> 3) as usize);
                        cache_length += 1;
                    }
                    ptr_inc!(data_ptr, BASE_SIZE);
                }
            }
            self.data_ptr = data_ptr;
            self.cache_length = cache_length;
            cache_length > 0 || self.fill_unaligned(self.data_end)
        }

        #[inline(always)]
        fn next(&mut self) -> *const u8 {
            let cache_offset = self.cache_offset;
            if self.cache_length > cache_offset {
                self.cache_offset += 1;
                self.cache[cache_offset]
            } else if self.fill() {
                self.cache_offset = 1;
                self.cache[0]
            } else {
                null()
            }
        }
    }

    struct Group<'a> {
        commas: Tokenizer<'a>,
        newlines: Tokenizer<'a>,
        leading: *const u8,
    }

    impl<'a> Group<'a> {
        fn new(data: &'a [u8]) -> Self {
            Self {
                leading: data.as_ptr(),
                commas: Tokenizer::new(data, CHR_CM, &FIND_MASK_CM),
                newlines: Tokenizer::new(data, CHR_NL, &FIND_MASK_NL),
            }
        }

        #[inline(always)]
        fn for_each<F>(&mut self, mut f: F) -> u64
        where
            F: FnMut((City<'a>, Temperature)),
        {
            let commas = &mut self.commas;
            let newlines = &mut self.newlines;
            let mut newline = self.leading;
            macro_rules! mid_slice {
                ($s:expr, $e:expr) => {
                    unsafe { slice::from_raw_parts($s, ($e as usize) - ($s as usize)) }
                };
            }
            let mut total = 0;
            macro_rules! null_check {
                ($e:expr) => {{
                    if !$e.is_null() {
                        $e
                    } else {
                        break total;
                    }
                }};
            }
            loop {
                let mut comma = commas.next();
                let city = { mid_slice!(newline, null_check!(comma)) };
                let value = {
                    ptr_inc!(comma, 1);
                    newline = newlines.next();
                    mid_slice!(comma, null_check!(newline))
                };
                ptr_inc!(newline, 1);
                f((city.into(), parse_number(value)));
                total += 1;
            }
        }
    }

    pub fn decode_lines_a<'a>(data: &'a [u8], result: &mut WeatherMap<'a>, dry_run: bool) -> u64 {
        let mut group = Group::new(data);
        if dry_run {
            group.for_each(drop)
        } else {
            group.for_each(
                #[inline(always)]
                move |v| result.put(v),
            )
        }
    }

    macro_rules! find_simd_mask {
        ($chr:expr, $mask: expr, $cache:expr, $data_ptr:expr, $end_ptr: expr) => {{
            // boost performance with simd
            let (mut count, last) = (0, ptr_add!($end_ptr, -(FIND_SIZE as isize)));
            while $data_ptr <= last {
                let value = $mask ^ read_unaligned!($data_ptr, FindSimd);
                for mut x in ((value - FIND_MASK1) & !value & FIND_MASK2).to_array() {
                    while x != 0 {
                        let v = x.trailing_zeros();
                        x ^= 1 << v;
                        $cache[count] = ptr_add!($data_ptr, (v >> 3) as usize);
                        count += 1;
                    }
                    ptr_inc!($data_ptr, BASE_SIZE);
                }
                if count != 0 {
                    return count;
                }
            }
            while $data_ptr < $end_ptr {
                if read!($data_ptr) == $chr {
                    $cache[count] = $data_ptr;
                    count += 1;
                }
                ptr_inc!($data_ptr, 1)
            }
            count
        }};
    }

    macro_rules! define_simd_find {
        ($name:ident, $chr: ident, $mask: ident) => {
            #[inline(always)]
            #[allow(unused_unsafe)]
            pub fn $name(
                cache: &mut [*const u8],
                mut data_ptr: *const u8,
                end_ptr: *const u8,
            ) -> usize {
                unsafe { find_simd_mask!($chr, $mask, cache, data_ptr, end_ptr) }
            }
        };
    }
    define_simd_find!(find_comma_simd, CHR_CM, FIND_MASK_CM);
    define_simd_find!(find_newline_simd, CHR_NL, FIND_MASK_NL);

    pub fn decode_lines_b<'a>(data: &'a [u8], result: &mut WeatherMap<'a>, dry_run: bool) -> u64 {
        fn for_each<'a, F>(data: &'a [u8], mut f: F) -> u64
        where
            F: FnMut((City<'a>, Temperature)),
        {
            macro_rules! mid_slice {
                ($s:expr, $e:expr) => {
                    unsafe { slice::from_raw_parts($s, ($e as usize) - ($s as usize)) }
                };
            }
            let mut commas = [null(); SIMD_SOLTS];
            let mut newlns = [null(); SIMD_SOLTS];
            let mut newline = data.as_ptr();
            let end = ptr_add!(newline, data.len());
            let mut total = 0u64;
            macro_rules! pipeline {
                ($i:expr) => {{
                    let mut comma = commas[$i];
                    let city = mid_slice!(newline, comma);
                    newline = newlns[$i];
                    ptr_inc!(comma, 1);
                    let value = mid_slice!(comma, newline);
                    ptr_inc!(newline, 1);
                    f((city.into(), parse_number(value)));
                }};
            }
            macro_rules! repeat {
                ($($i:expr)*) => {
                    $(pipeline!($i);)*
                };
            }
            loop {
                let mut count = find_comma_simd(&mut commas, newline, end);
                count = count.min(find_newline_simd(&mut newlns, ptr_add!(commas[0], 1), end));
                total += match count {
                    0 => {
                        break total;
                    }
                    1 => {
                        repeat!(0);
                        1
                    }
                    _ => {
                        repeat!(0 1);
                        2
                    }
                };
            }
        }
        if dry_run {
            for_each(data, drop)
        } else {
            for_each(
                data,
                #[inline(always)]
                |v| result.put(v),
            )
        }
    }

    macro_rules! find_mask {
        ($chr:expr, $mask: expr, $data_ptr:expr, $end_ptr: expr) => {{
            // boost performance with swar
            let last = ptr_add!($end_ptr, -(FIND_SIZE as isize));
            while $data_ptr <= last {
                let mut value = $mask ^ read_unaligned!($data_ptr, FindBase);
                value = (value - BASE_MASK1) & !value & BASE_MASK2;
                if value != 0 {
                    return ptr_add!($data_ptr, (value.trailing_zeros() >> 3) as usize);
                }
                ptr_inc!($data_ptr, BASE_SIZE);
            }
            while $data_ptr < $end_ptr {
                if read!($data_ptr) == $chr {
                    return $data_ptr;
                }
                ptr_inc!($data_ptr, 1);
            }
            null()
        }};
    }

    macro_rules! define_find {
        ($name:ident, $chr: ident, $mask: ident) => {
            #[inline(always)]
            pub fn $name(mut data_ptr: *const u8, end_ptr: *const u8) -> *const u8 {
                find_mask!($chr, $mask, data_ptr, end_ptr)
            }
        };
    }

    // define_find!(find_comma, CHR_CM, BASE_MASK_CM);
    define_find!(find_newline, CHR_NL, BASE_MASK_NL);

    #[allow(unused_unsafe)]
    pub fn decode_lines_c<'a>(data: &'a [u8], result: &mut WeatherMap<'a>, dry_run: bool) -> u64 {
        fn for_each<'a, F>(data: &'a [u8], mut f: F) -> u64
        where
            F: FnMut((City<'a>, Temperature)),
        {
            let mut head = data.as_ptr();
            let mut mark = head;
            let end = ptr_add!(head, data.len());
            let last = ptr_add!(end, -((BASE_SIZE * 8) as isize));
            let mut total = 0u64;
            #[cfg(feature = "hit_miss")]
            let mut miss = 0;
            'next: while head <= last {
                macro_rules! find {
                    ($mask: expr, $input: expr) => {{
                        let value = $mask ^ $input;
                        (value - BASE_MASK1) & !value & BASE_MASK2
                    }};
                }
                let wordx = FindBase::to_le(read_unaligned!(head, FindBase));
                macro_rules! step {
                    () => {
                        step!(
                            0,
                            wordx,
                            FindBase::to_le(read_unaligned!(head, BASE_SIZE, FindBase))
                        );
                    };
                    ($i: expr) => {
                        step!(
                            $i,
                            FindBase::to_le(read_unaligned!(head, $i, FindBase)),
                            FindBase::to_le(read_unaligned!(head, $i + BASE_SIZE, FindBase))
                        );
                    };
                    ($i: expr, $word1: expr, $word2: expr) => {{
                        let tz0 = find!(BASE_MASK_CM, $word1);
                        let tz1 = find!(BASE_MASK_CM, $word2);

                        if (tz0 | tz1) != 0 {
                            let i = (tz0 != 0) as usize;
                            let len = (head as usize - mark as usize)
                                + [$i + BASE_SIZE, $i][i]
                                + ([tz1, tz0][i].trailing_zeros() >> 3) as usize;
                            let city =
                                City::new_ex(unsafe { slice::from_raw_parts(mark, len) }, wordx);
                            ptr_inc!(mark, len + 1);
                            let value = FindBase::to_le(read_unaligned!(mark, FindBase));
                            let len = find!(BASE_MASK_NL, value).trailing_zeros() >> 3;
                            let value = parse_number_ex(value, len);
                            ptr_inc!(mark, len + 1);
                            head = mark;
                            f((city, value));
                            total += 1;
                            continue 'next;
                        }
                        #[cfg(feature = "hit_miss")]
                        {
                            miss += 1;
                        }
                    }};
                }
                step!();
                step!(BASE_SIZE * 2);
                step!(BASE_SIZE * 4);
                step!(BASE_SIZE * 6);
                ptr_inc!(head, BASE_SIZE * 8);
            }
            'next: while head < end {
                macro_rules! slice {
                    () => {
                        unsafe {
                            let r = slice::from_raw_parts(mark, head as usize - mark as usize);
                            ptr_inc!(head, 1);
                            mark = head;
                            r
                        }
                    };
                }
                match read!(head) {
                    CHR_CM => {
                        let city = slice!();
                        while head < end {
                            match read!(head) {
                                CHR_NL => {
                                    let value = slice!();
                                    f((city.into(), parse_number(value)));
                                    total += 1;
                                    continue 'next;
                                }
                                _ => ptr_inc!(head, 1),
                            }
                        }
                    }
                    _ => ptr_inc!(head, 1),
                }
            }
            #[cfg(feature = "hit_miss")]
            {
                eprintln!(
                    "{:?} {:.3}%",
                    thread::current().id(),
                    100f64 * miss as f64 / total as f64
                );
            }
            total
        }
        if dry_run {
            for_each(data, drop)
        } else {
            for_each(
                data,
                #[inline(always)]
                |v| result.put(v),
            )
        }
    }

    #[inline(always)]
    fn parse_number(value: &[u8]) -> Temperature {
        let sign = (read!(value.as_ptr()) == b'-') as u32;
        let value = u32::to_le(read_unaligned!(value.as_ptr(), sign, u32))
            << ((sign + 4 - value.len() as u32) << 3);
        parse_number_magic(value, sign)
    }

    #[inline(always)]
    fn parse_number_ex(value: u64, len: u32) -> Temperature {
        let sign = (value as u8 == b'-') as u32;
        let value = ((value >> sign << 3) as u32) << ((sign + 4 - len) << 3);
        parse_number_magic(value, sign)
    }

    #[inline(always)]
    fn parse_number_magic(value: u32, sign: u32) -> Temperature {
        macro_rules! sign {
            ($v:expr, $i: expr) => {
                (($v) as i64 ^ -(($i) as i64)) + $i as i64
            };
        }
        sign!(
            ((((value & 0x0F000F0F) as u64).wrapping_mul(0x640A000100) >> 32) & 0x3FF),
            sign
        ) as Temperature
    }

    #[cfg(test)]
    pub mod tests {
        use crate::{
            bench::{HIS_MISS, WeatherMap, decode_lines_c as decode_lines, parse_number},
            r#gen::Mmap,
        };
        use clap::Parser;
        use std::fs::File;
        extern crate test;

        #[derive(Parser)]
        #[command(version, about)]
        struct BenchArg {
            /// data file
            #[arg(long, default_value = "./data/measurements.txt")]
            data: String,

            /// dry-run without map/reduce
            #[arg(short, long)]
            dry_run: bool,

            /// data file in hugetlbfs
            #[arg(long)]
            hugepages: bool,
        }

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
            let data = "aaaaaaaaa;-10.0\naaaaaaaaa;26.0\ndef;2.1\n".as_bytes();
            let mut m = WeatherMap::default();
            decode_lines(data, &mut m, false);
            assert_eq!(m.len(), 2);
            let r = m.get(&"aaaaaaaaa".as_bytes().into()).unwrap();
            assert_eq!(r.cnt, 2);
            assert_eq!(r.min, -100);
            assert_eq!(r.max, 260);
            assert_eq!(r.sum, 160);
        }

        #[test]
        pub fn test_swar() {
            fn fmt(mut v: String) -> String {
                let mut i = v.len() - 8;
                while i > 0 {
                    v.insert(i, ' ');
                    i -= 8;
                }
                v
            }
            macro_rules! e {
                ($b: expr, $e:expr, $x: expr) => {
                    $b.push_str(&format!(
                        "{:>32} | {} | {}\n",
                        stringify!($e),
                        fmt(format!("{:032b}", $e)),
                        $x
                    ))
                };
                ($b: expr, $e:expr) => {
                    $b.push_str(&format!("{:>32} | {}\n", stringify!($e), $e));
                };
            }
            // boost performance with swar
            const SIZE: usize = size_of::<u32>();
            const MASK: u32 = u32::from_ne_bytes([b';'; SIZE]);
            const MASK1: u32 = u32::from_ne_bytes([0x01; SIZE]);
            const MASK2: u32 = u32::from_ne_bytes([0x80; SIZE]);
            let mut arr = [b'1'; SIZE];
            arr[SIZE - 2] = b';';
            let mut value = u32::from_ne_bytes(arr);
            let mut buf = String::new();

            e!(buf, value, "");
            e!(buf, MASK, "");
            e!(buf, value ^ MASK, "=>value");
            value ^= MASK;
            e!(buf, MASK1, "");
            e!(buf, (value - MASK1), "");
            e!(buf, !value, "");
            e!(buf, (value - MASK1) & !value, "");
            e!(buf, MASK2, "");
            e!(buf, (value - MASK1) & !value & MASK2, "");
            value = (value - MASK1) & !value & MASK2;
            e!(buf, value.trailing_zeros());
            e!(buf, value.trailing_zeros() >> 3);
            assert_eq!(value.trailing_zeros() >> 3, (SIZE - 2) as u32);

            eprintln!("\n\n{}\n", buf.as_str());
        }

        #[bench]
        #[ignore]
        fn bench_reduce(b: &mut test::Bencher) {
            let cli = BenchArg::parse();
            let data = Mmap::open::<false>(File::open(cli.data).unwrap(), cli.hugepages).unwrap();
            let mut m = WeatherMap::default();
            b.iter(|| {
                HIS_MISS.with(|x| {
                    x.set(0);
                    decode_lines(&data, m.reset(), cli.dry_run);
                });
            });
        }

        #[bench]
        #[ignore]
        fn bench_mmap(b: &mut test::Bencher) {
            let cli = BenchArg::parse();
            let data = Mmap::open::<false>(File::open(cli.data).unwrap(), cli.hugepages).unwrap();
            b.iter(|| {
                let len = data.len();
                let mut data_ptr = data.as_ptr();
                let data_end = unsafe { data_ptr.add(len) };
                let data_last = unsafe { data_ptr.add(len - 8) };
                let mut x = 0;
                while data_ptr < data_last {
                    x ^= unsafe { *data_ptr.cast::<u64>() };
                    data_ptr = unsafe { data_ptr.add(size_of::<u64>()) };
                }
                while data_ptr < data_end {
                    x ^= unsafe { *data_ptr } as u64;
                    data_ptr = unsafe { data_ptr.add(1) };
                }
                println!("{x}");
            });
        }
    }
}
