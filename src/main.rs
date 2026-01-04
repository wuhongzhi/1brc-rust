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
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write, stdout},
    mem::ManuallyDrop,
    os::fd::FromRawFd,
    process::exit,
    time::SystemTime,
};

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
            slice::from_raw_parts($d.as_ptr().add(off), len)
        }
    };
}
macro_rules! mid_slice {
    ($d:expr, $s:expr, $e:expr) => {
        unsafe { slice::from_raw_parts($d.as_ptr().add($s), $e - ($s)) }
    };
}
macro_rules! read_byte {
    ($e:expr) => {
        unsafe { *$e }
    };
    ($e:expr, $i:expr) => {
        unsafe { *$e.add($i) }
    };
}
const RESULT_BITS: u32 = 16;

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
    #[arg(short, long, default_value_t = 8_000, value_parser = max_cities)]
    cities: usize,

    /// template file
    #[arg(short, long, default_value = "./data/weather_stations.csv")]
    template: String,

    /// data file
    #[arg(short, long, default_value = "./data/measurements.txt")]
    data: String,

    /// write data line by line
    #[arg(short, long)]
    legacy: bool,
}
fn max_cities(s: &str) -> Result<usize> {
    Ok(s.parse::<usize>()?.clamp(1, 10_000))
}
#[derive(Args)]
struct BenchArg {
    /// data file
    #[arg(short, long, default_value = "./data/measurements.txt")]
    data: String,

    /// slice size, default to file size / workers
    #[arg(short, long)]
    slice: Option<usize>,

    /// parallel workers, default to cpu cores
    #[arg(short, long)]
    workers: Option<usize>,

    /// dry-run without map/reduce
    #[arg(long)]
    dry_run: bool,
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Gen(g) => generate(g),
        Cli::Bench(b) => bench(b),
    }
}
fn generate(argv: GenerateArg) -> Result<()> {
    let data = {
        let mut data = String::new();
        let mut file = File::open(argv.template)?;
        file.read_to_string(&mut data)?;
        data
    };
    let mut cities: Vec<(bench::City<'_>, bool)> = {
        let mut cities = HashSet::new();
        data.lines().for_each(|line| {
            if !line.starts_with("#")
                && let Some(city) = line.split(";").next()
            {
                cities.insert(bench::City::new(city.as_bytes()));
            }
        });
        cities
            .into_iter()
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
        macro_rules! rand {
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
                    $e.write_all(&city.0)?;
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

    eprintln!(
        "Final size {} {} with {} cities",
        ((len * 100f64).round() / 100f64).format(3),
        units[i],
        cities.iter().filter(|f| f.1).count().format(0)
    );
    Ok(())
}

fn bench(argv: BenchArg) -> Result<()> {
    let clock = SystemTime::now();
    let mut pipe_fds = [0i32; 2];
    unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
    match fork()? {
        Fork::Parent(_child) => {
            unsafe {
                libc::fcntl(pipe_fds[0], libc::F_SETPIPE_SZ, 1 << 20);
                libc::close(pipe_fds[1]);
            }
            let mut buf = new_buf();
            let mut reader = unsafe { File::from_raw_fd(pipe_fds[0]) };
            reader.read_to_end(&mut buf)?;
            stdout().write_all(&buf)?;
        }
        Fork::Child => match Mmap::open::<false>(File::open(argv.data)?) {
            Ok(data) => {
                bench::reduce(&data, argv.slice, argv.workers, argv.dry_run)?
                    .write(unsafe { File::from_raw_fd(pipe_fds[1]) }, new_buf())?
                    .wait(clock)?;
            }
            Err(e) => {
                eprintln!("{e:?}");
            }
        },
    }
    exit(0);
    fn new_buf() -> ManuallyDrop<Vec<u8>> {
        ManuallyDrop::new(Vec::with_capacity(512_000))
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
    }

    impl Mmap {
        pub fn open<const WRITE: bool>(file: File) -> Result<Self> {
            let length = file.metadata()?.len() as _;
            let write = WRITE;
            if write {
                let mut chunk = 64 * 1024;
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
                if length == 0 {
                    bail!("Empty file");
                }
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
    use crate::RESULT_BITS;
    use anyhow::{Ok, Result};
    use concat_idents::concat_idents;
    use core::slice;
    use proc_cpuinfo::CpuInfo;
    use rapidhash::fast::RandomState;
    use rayon::prelude::*;
    use std::{
        collections::BTreeMap,
        fmt::Display,
        fs::File,
        hash::{BuildHasher, Hash, Hasher},
        io::Write,
        iter::FusedIterator,
        mem::ManuallyDrop,
        ops::{AddAssign, Deref},
        ptr,
        simd::{u8x8, u16x8, u32x8, u64x8},
        thread,
        time::SystemTime,
    };

    #[derive(Clone, Copy)]
    pub struct Weather {
        min: i16,
        max: i16,
        sum: i64,
        count: u64,
    }
    impl Weather {
        fn new(value: i16) -> Self {
            Self {
                min: value,
                max: value,
                sum: value as i64,
                count: 1,
            }
        }
        fn write(&self, buf: &mut Vec<u8>) {
            let mut avg = (self.sum as f64 / self.count as f64).round();
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
    impl From<i16> for Weather {
        fn from(value: i16) -> Self {
            Self::new(value)
        }
    }
    impl AddAssign for Weather {
        fn add_assign(&mut self, other: Self) {
            self.sum += other.sum;
            self.count += other.count;
            self.max = self.max.max(other.max);
            self.min = self.min.min(other.min);
        }
    }
    macro_rules! read_unaligned {
        ($p:expr, $ty:ty) => {
            unsafe { ptr::read_unaligned($p as *const $ty) }
        };
    }
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialOrd, Ord, Eq)]
    pub struct City<'a> {
        pub name: &'a [u8],
    }
    impl<'a> City<'a> {
        pub fn new(name: &'a [u8]) -> Self {
            Self { name }
        }
        pub fn write(&self, buf: &mut Vec<u8>) {
            buf.extend_from_slice(self.name);
        }
    }
    impl<'a> From<&'a [u8]> for City<'a> {
        fn from(value: &'a [u8]) -> Self {
            Self::new(value)
        }
    }
    impl<'a> Hash for City<'a> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // self.name.hash(state);
            let a = self.name;
            match a.len() >> 4 {
                0 => a.hash(state),
                _ => {
                    let ptr = a.as_ptr();
                    read_unaligned!(ptr, u64).hash(state);
                    read_unaligned!(ptr.add(a.len() - size_of::<u64>()), u64).hash(state);
                }
            }
        }
    }
    impl<'a> PartialEq for City<'a> {
        fn eq(&self, other: &Self) -> bool {
            // self.name.eq(other.name)
            #[inline(always)]
            fn slow_loop(a: &[u8], b: &[u8]) -> bool {
                (0..a.len()).all(|i| read_byte!(a.as_ptr(), i) == read_byte!(b.as_ptr(), i))
            }
            #[inline(always)]
            #[allow(unused_unsafe)]
            fn fast_simd(a: &[u8], b: &[u8]) -> bool {
                macro_rules! cast_x8 {
                    ($ty:expr, $e: expr) => {
                        concat_idents!(ty = u, $ty {
                            unsafe { slice::from_raw_parts::<ty>($e.as_ptr().cast(), 8) }
                        })
                    };
                }
                macro_rules! simd_ne {
                    ($w:expr, $a:expr, $b:expr) => {
                        concat_idents!(simd = u, $w, x8 {
                            <simd>::from_slice(cast_x8!($w, $a))
                                != <simd>::from_slice(cast_x8!($w, $b))
                        })
                    };
                    ($c:expr, $d:expr, $a:expr, $b:expr) => {
                        simd_ne!($c, $a, $b)
                            || simd_ne!($d, pst_slice!($a, $c), pst_slice!($b, $c))
                    };
                    ($c:expr, $d:expr, $e:expr, $a:expr, $b:expr) => {
                        simd_ne!($c, $d, $a, $b)
                            || simd_ne!(
                                $e,
                                pst_slice!($a, $c + $d),
                                pst_slice!($b, $c + $d)
                            )
                    };
                }
                match a.len() >> 3 {
                    1 if simd_ne!(8, a, b) => false,
                    2 if simd_ne!(16, a, b) => false,
                    3 if simd_ne!(16, 8, a, b) => false,
                    4 if simd_ne!(32, a, b) => false,
                    5 if simd_ne!(32, 8, a, b) => false,
                    6 if simd_ne!(32, 16, a, b) => false,
                    7 if simd_ne!(32, 16, 8, a, b) => false,
                    8 if simd_ne!(64, a, b) => false,
                    x @ 0..=8 => slow_loop(pst_slice!(a, x << 3), pst_slice!(b, x << 3)),
                    _ => a == b,
                }
            }

            #[inline(always)]
            fn fast_length(a: usize, b: usize) -> bool {
                a != b
            }
            #[inline(always)]
            //fast detect first & last 2bytes base on the city statistics
            fn fast_detect(a: &[u8], b: &[u8]) -> bool {
                read_byte!(a.as_ptr()) != read_byte!(b.as_ptr())
                    || read_unaligned!(a.as_ptr().add(a.len() - 2), u16)
                        != read_unaligned!(b.as_ptr().add(b.len() - 2), u16)
            }

            let a = self.name;
            let b = other.name;
            let len = a.len();

            if fast_length(len, b.len()) {
                false
            } else if len < 4 {
                slow_loop(a, b)
            } else if fast_detect(a, b) {
                false
            } else {
                fast_simd(mid_slice!(a, 1, len - 2), mid_slice!(b, 1, len - 2))
            }
        }
    }
    impl<'a> Deref for City<'a> {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            self.name
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
                    find_newline(pst_slice!(self.data, expect_end))
                        .map_or(self_end, |i| expect_end + i + 1)
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

    #[derive(Clone, Copy)]
    enum MyWeatherNode<'a> {
        Value((City<'a>, Weather)),
        Empty,
    }

    pub struct MyWeatherMap<'a> {
        inner: Vec<MyWeatherNode<'a>>,
        hasher: RandomState,
    }

    const BITS_MASK: usize = (1 << RESULT_BITS) - 1;

    impl<'a> Default for MyWeatherMap<'a> {
        fn default() -> Self {
            MyWeatherMap {
                inner: vec![MyWeatherNode::Empty; 1 << RESULT_BITS],
                hasher: RandomState::default(),
            }
        }
    }

    macro_rules! set {
        ($v:expr, $r:expr) => {
            unsafe { *$v = $r };
        };
        ($v:expr, $i:expr, $r:expr) => {
            unsafe { *$v.add($i) = $r };
        };
    }
    macro_rules! get {
        ($v:expr) => {
            unsafe { *$v }
        };
        ($v:expr, $i:expr) => {
            unsafe { *$v.add($i) }
        };
    }
    #[allow(dead_code)]
    impl<'a> MyWeatherMap<'a> {
        pub fn reset(&mut self) -> &mut Self {
            self.inner.iter_mut().for_each(|node| {
                if matches!(node, MyWeatherNode::Value(_)) {
                    *node = MyWeatherNode::Empty;
                }
            });
            self
        }
        pub fn len(&self) -> usize {
            self.inner
                .iter()
                .filter(|x| matches!(x, MyWeatherNode::Value(_)))
                .count()
        }
        pub fn get(&self, city: &City<'static>) -> Option<&Weather> {
            for i in self.inner.iter() {
                if let MyWeatherNode::Value(v) = i
                    && v.0.eq(city)
                {
                    return Some(&v.1);
                }
            }
            None
        }
        pub fn iter(self) -> WeatherIter<'a> {
            WeatherIter {
                pos: 0,
                inner: self.inner,
            }
        }

        // TODO: refine this method
        #[inline(always)]
        pub fn put<const MASK: usize>(&mut self, (key, value): (City<'a>, i16)) {
            #[inline(always)]
            fn index(index: u64) -> usize {
                (/* index.rotate_right(RESULT_BITS * 4)
                ^  */(index >> (RESULT_BITS * 3))
                    ^ (index >> (RESULT_BITS * 2))
                    ^ (index >> RESULT_BITS)
                    ^ index) as usize
            }
            let ptr = self.inner.as_mut_ptr();
            let (mut miss, mut index) = (0, index(self.hasher.hash_one(key)) & MASK);
            loop {
                match unsafe { &mut *ptr.add(index) } {
                    MyWeatherNode::Value((city, weather)) => {
                        if key.eq(city) {
                            weather.count += 1;
                            weather.sum += value as i64;
                            weather.max = weather.max.max(value);
                            weather.min = weather.min.min(value);
                            break;
                        }
                        miss += 1;
                        assert!(miss <= BITS_MASK, "Map is full!");
                        index = (index + 9337) & MASK;
                    }
                    node => {
                        *node = MyWeatherNode::Value((key, value.into()));
                        break;
                    }
                }
            }
        }
    }
    pub struct WeatherIter<'a> {
        pos: usize,
        inner: Vec<MyWeatherNode<'a>>,
    }
    impl<'a> FusedIterator for WeatherIter<'a> {}
    impl<'a> Iterator for WeatherIter<'a> {
        type Item = (City<'a>, Weather);

        fn next(&mut self) -> Option<Self::Item> {
            let ptr = self.inner.as_mut_ptr();
            while self.pos < self.inner.len() {
                if let MyWeatherNode::Value(x) = unsafe { &mut *ptr.add(self.pos) } {
                    self.pos += 1;
                    return Some(*x);
                }
                self.pos += 1;
            }
            None
        }
    }

    #[repr(transparent)]
    pub struct Reduce<'a>((BTreeMap<City<'a>, Weather>, u64));

    impl<'a> Reduce<'a> {
        pub fn write(self, mut file: File, mut buf: ManuallyDrop<Vec<u8>>) -> Result<Self> {
            buf.push(b'{');
            self.0
                .0
                .iter()
                .enumerate()
                .for_each(|(id, (city, weather))| {
                    if id != 0 {
                        buf.extend_from_slice(", ".as_bytes());
                    }
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
            eprintln!(
                "Result in {}ms with {} lines and {} cities, on average of {:.3}ns/line",
                (taken.as_micros() as f64 / 1_000f64).format(3),
                self.0.1.format(0),
                self.0.0.len().format(0),
                taken.as_nanos() as f64 / self.0.1 as f64
            );
            Ok(())
        }
    }

    pub fn reduce(
        data: &[u8],
        slice: Option<usize>,
        workers: Option<usize>,
        dry_run: bool,
    ) -> Result<Reduce<'_>> {
        fn map<'a>(dry_run: bool) -> impl Fn(&'a [u8]) -> (BTreeMap<City<'a>, Weather>, u64) {
            move |part| {
                let clock = SystemTime::now();
                let mut cities = WeatherMap::default();
                let total = decode_lines(part, &mut cities, dry_run);
                if dry_run {
                    eprintln!(
                        "{:?} -> decode {} lines within {}ms",
                        thread::current().id(),
                        total.format(0),
                        (clock.elapsed().unwrap().as_micros() as f64 / 1_000f64).format(3)
                    );
                }
                let mut result: BTreeMap<City<'_>, Weather> = BTreeMap::default();
                result.extend(cities.iter());
                (result, total)
            }
        }
        fn reduce<'a>(
            mut result: (BTreeMap<City<'a>, Weather>, u64),
            cities: (BTreeMap<City<'a>, Weather>, u64),
        ) -> (BTreeMap<City<'a>, Weather>, u64) {
            cities.0.into_iter().for_each(|(city, value)| {
                result
                    .0
                    .entry(city)
                    .and_modify(|weather| *weather += value)
                    .or_insert(value);
            });
            result.1 += cities.1;
            result
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
        let scanner = Scanner::new(
            unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) },
            slice
                .unwrap_or(data.len() / cpu_cores)
                .max(cache_size / cpu_cores),
        );
        Ok(Reduce(
            match cpu_cores {
                1 => scanner.map(map(dry_run)).reduce(reduce),
                _ => {
                    rayon::ThreadPoolBuilder::new()
                        .thread_name(|i| format!("decode-worker-{i}"))
                        .num_threads(cpu_cores)
                        .use_current_thread()
                        .build_global()?;
                    scanner
                        .par_bridge()
                        .into_par_iter()
                        .map(map(dry_run))
                        .reduce_with(reduce)
                }
            }
            .unwrap(),
        ))
    }

    const CHR_NL: u8 = b'\n';
    const CHR_CM: u8 = b';';
    type FindBase = u64;
    type FindSimd = std::simd::u64x4;
    //min 7 byets one solt (ab;y.z\n)
    const SIMD_SOLTS: usize = (size_of::<FindSimd>() as f32 / 7f32).ceil() as usize;

    const BASE_SIZE: usize = size_of::<FindBase>();
    const FIND_SIZE: usize = size_of::<FindSimd>();
    const FIND_MASK1: FindSimd = FindSimd::splat(FindBase::from_ne_bytes([0x01; BASE_SIZE]));
    const FIND_MASK2: FindSimd = FindSimd::splat(FindBase::from_ne_bytes([0x80; BASE_SIZE]));

    #[inline(always)]
    pub fn find_newline(data: &[u8]) -> Option<usize> {
        const MASK: FindBase = FindBase::from_ne_bytes([CHR_NL; BASE_SIZE]);
        const MASK1: FindBase = FindBase::from_ne_bytes([0x01; BASE_SIZE]);
        const MASK2: FindBase = FindBase::from_ne_bytes([0x80; BASE_SIZE]);
        let (mut off, end) = (0, data.len());
        let ptr = data.as_ptr();
        // boost performance with swar
        for _ in 0..end / BASE_SIZE {
            let mut value = read_unaligned!(ptr.add(off), FindBase) ^ MASK;
            value = (value - MASK1) & !value & MASK2;
            if value != 0 {
                return Some(off + (value.trailing_zeros() >> 3) as usize);
            }
            off += BASE_SIZE;
        }
        for j in off..end {
            if read_byte!(ptr, j) == CHR_NL {
                return Some(j);
            }
        }
        None
    }

    // struct Tokenizer<'a> {
    //     cache: [usize; SIMD_SOLTS],
    //     cache_offset: usize,
    //     cache_length: usize,
    //     data: &'a [u8],
    //     data_offset: usize,
    //     data_align: usize,
    //     chr: u8,
    //     mask: usize,
    // }
    // impl<'a> Tokenizer<'a> {
    //     fn new(data: &'a [u8], chr: u8) -> Self {
    //         Self {
    //             data,
    //             chr,
    //             cache: [0; SIMD_SOLTS],
    //             cache_length: 0,
    //             cache_offset: 0,
    //             data_offset: 0,
    //             data_align: 0,
    //             mask: if chr == CHR_CM { 0 } else { 1 },
    //         }
    //         .align()
    //     }

    //     #[inline(always)]
    //     fn align(mut self) -> Self {
    //         let end = self.data.len();
    //         if self.data_offset < end {
    //             self.data_align = match self.data.as_ptr().addr() % FIND_SIZE {
    //                 0 => end / FIND_SIZE * FIND_SIZE,
    //                 mut unaligned => {
    //                     let mut count = 0;
    //                     unaligned = (FIND_SIZE - unaligned).min(end);
    //                     for i in 0..unaligned {
    //                         if read_byte!(self.data.as_ptr(), i) == self.chr {
    //                             set!(self.cache.as_mut_ptr(), count, i);
    //                             count += 1;
    //                         }
    //                     }
    //                     if count != 0 {
    //                         self.cache_offset = 0;
    //                         self.cache_length = count;
    //                     }
    //                     self.data_offset = unaligned;
    //                     (end - unaligned) / FIND_SIZE * FIND_SIZE
    //                 }
    //             };
    //         }
    //         self
    //     }

    //     #[inline(always)]
    //     fn fill(&mut self) -> bool {
    //         const MASK: [FindSimd; 2] = [
    //             FindSimd::splat(FindBase::from_ne_bytes([CHR_CM; BASE_SIZE])),
    //             FindSimd::splat(FindBase::from_ne_bytes([CHR_NL; BASE_SIZE])),
    //         ];
    //         let mut count = 0;
    //         macro_rules! check_result {
    //             ($delta:expr) => {
    //                 self.data_offset += $delta;
    //                 if count != 0 {
    //                     self.cache_offset = 0;
    //                     self.cache_length = count;
    //                     return true;
    //                 }
    //             };
    //         }
    //         while self.data_offset < self.data_align {
    //             let mut value = get!(MASK.as_ptr(), self.mask)
    //                 ^ get!(self.data.as_ptr().add(self.data_offset).cast::<FindSimd>());
    //             value = (value - FIND_MASK1) & !value & FIND_MASK2;
    //             let value = value.as_array();
    //             let mut off = self.data_offset;
    //             for j in 0..FIND_SIZE / BASE_SIZE {
    //                 let mut x = get!(value.as_ptr(), j);
    //                 while x != 0 {
    //                     let v = x.trailing_zeros();
    //                     set!(self.cache.as_mut_ptr(), count, off + (v >> 3) as usize);
    //                     x ^= 1 << v;
    //                     count += 1;
    //                 }
    //                 off += BASE_SIZE;
    //             }
    //             check_result!(FIND_SIZE);
    //         }
    //         for j in self.data_offset..self.data.len() {
    //             if read_byte!(self.data.as_ptr(), j) == self.chr {
    //                 set!(self.cache.as_mut_ptr(), count, j);
    //                 count += 1;
    //             }
    //         }
    //         check_result!(self.data.len() - self.data_offset);
    //         false
    //     }

    //     #[inline(always)]
    //     fn next(&mut self) -> (usize, usize) {
    //         macro_rules! available {
    //             () => {
    //                 self.cache_length - self.cache_offset
    //             };
    //         }
    //         if available!() > 0 || self.fill() {
    //             let r = get!(self.cache.as_ptr(), self.cache_offset);
    //             self.cache_offset += 1;
    //             (r, available!())
    //         } else {
    //             (0, 0)
    //         }
    //     }

    //     #[inline(always)]
    //     fn take(&mut self) -> usize {
    //         let r = get!(self.cache.as_ptr(), self.cache_offset);
    //         self.cache_offset += 1;
    //         r
    //     }
    // }

    // #[allow(unused_unsafe)]
    // pub fn decode_lines<'a>(data: &'a [u8], result: &mut WeatherMap<'a>, dry_run: bool) -> u64 {
    //     let mut commas = Tokenizer::new(data, CHR_CM);
    //     let mut newlines = Tokenizer::new(data, CHR_NL);
    //     let (mut leading, mut total) = (0, 0u64);
    //     while let (mut comma, c1) = commas.next()
    //         && let (mut newline, c2) = newlines.next()
    //         && comma != 0
    //         && newline != 0
    //     {
    //         macro_rules! pipeline {
    //             ($comma:expr, $newline: expr) => {{
    //                 let city = mid_slice!(data, leading, $comma);
    //                 let value = mid_slice!(data, $comma + 1, $newline);
    //                 leading = $newline + 1;
    //                 (city.into(), parse_number(value))
    //             }};
    //         }
    //         let v0 = pipeline!(comma, newline);
    //         macro_rules! repeat {
    //             ($($i: expr)*) => {{
    //                 $(
    //                     concat_idents!(name = v, $i {
    //                         comma = commas.take();
    //                         newline = newlines.take();
    //                         let name = pipeline!(comma, newline);
    //                     });
    //                 )*
    //                 if !dry_run {
    //                     result.put::<BITS_MASK>(v0);
    //                     $(
    //                         concat_idents!(name = v, $i {
    //                             result.put::<BITS_MASK>(name);
    //                         });
    //                     )*
    //                 }
    //             }};
    //         }
    //         total += match c2.min(c1) {
    //             x if x > 2 => {
    //                 repeat!(1 2 3);
    //                 4
    //             }
    //             x if x > 0 => {
    //                 repeat!(1);
    //                 2
    //             }
    //             _ => {
    //                 repeat!();
    //                 1
    //             }
    //         }
    //     }
    //     total
    // }

    macro_rules! find_mask {
        ($chr:expr, $data:expr, $pos_ptr:expr, $leading:expr) => {{
            const MASK: FindSimd = FindSimd::splat(FindBase::from_ne_bytes([$chr; BASE_SIZE]));
            let (mut count, mut leading, end) = (0, $leading, $data.len());
            // boost performance with simd
            for _ in 0..(end - leading) / FIND_SIZE {
                let mut value = read_unaligned!($data.as_ptr().add(leading), FindSimd) ^ MASK;
                value = (value - FIND_MASK1) & !value & FIND_MASK2;
                let value = value.as_array();
                let mut off = leading;
                for j in 0..FIND_SIZE / BASE_SIZE {
                    let mut x = get!(value.as_ptr(), j);
                    while x != 0 {
                        let v = x.trailing_zeros();
                        set!($pos_ptr, count, off + (v >> 3) as usize);
                        x ^= 1 << v;
                        count += 1;
                    }
                    off += BASE_SIZE;
                }
                match count {
                    0 => leading += FIND_SIZE,
                    x => return Some(x),
                }
            }
            for j in leading..end {
                if read_byte!($data.as_ptr(), j) == $chr {
                    set!($pos_ptr, count, j);
                    count += 1;
                }
            }
            match count {
                0 => None,
                x => Some(x),
            }
        }};
    }

    macro_rules! define_find {
        ($name:ident, $chr: ident) => {
            #[inline(always)]
            #[allow(unused_unsafe)]
            pub fn $name(data: &[u8], pos_ptr: *mut usize, leading: usize) -> Option<usize> {
                find_mask!($chr, data, pos_ptr, leading)
            }
        };
    }
    define_find!(find_comma_simd, CHR_CM);
    define_find!(find_newline_simd, CHR_NL);

    #[allow(unused_unsafe)]
    pub fn decode_lines<'a>(data: &'a [u8], result: &mut WeatherMap<'a>, dry_run: bool) -> u64 {
        let mut commas = [0; SIMD_SOLTS];
        let mut newlns = [0; SIMD_SOLTS];
        let cma_ptr = commas.as_mut_ptr();
        let nls_ptr = newlns.as_mut_ptr();
        let (mut leading, mut total) = (0, 0u64);
        while let Some(c1) = find_comma_simd(data, cma_ptr, leading)
            && let Some(c2) = find_newline_simd(data, nls_ptr, get!(cma_ptr) + 1)
        {
            macro_rules! pipeline {
                ($i:expr) => {{
                    let comma = get!(cma_ptr, $i);
                    let newline = get!(nls_ptr, $i);
                    let city = mid_slice!(data, leading, comma);
                    let value = mid_slice!(data, comma + 1, newline);
                    leading = newline + 1;
                    (city.into(), parse_number(value))
                }};
            }
            macro_rules! repeat {
                ($($i:expr)*) => {
                    $(
                        concat_idents!(name = v, $i {
                            let name = pipeline!($i);
                        });
                    )*
                    if !dry_run {
                    $(
                        concat_idents!(name = v, $i {
                            result.put::<BITS_MASK>(name);
                        });
                    )*
                    }
                };
            }
            total += match c2.min(c1) {
                x if x > 3 => {
                    repeat!(0 1 2 3);
                    4
                }
                x if x > 1 => {
                    repeat!(0 1);
                    2
                }
                _ => {
                    repeat!(0);
                    1
                }
            };
        }
        total
    }

    #[inline(always)]
    fn parse_number(value: &[u8]) -> i16 {
        let p = value.as_ptr();
        // signal bit 001(0)1101 => `-`
        let s = ((!read_byte!(p) & 0x10) >> 4) as usize;
        // boost performance with swar
        let v = u32::from_be(read_unaligned!(p.add(s), u32) & 0x0F0F0F0F)
            >> ((s + 4 - value.len()) << 3);
        (100 * (v >> 24) + 10 * ((v << 8) >> 24) + ((v << 24) >> 24)) as i16 * (1 - (s << 1) as i16)
    }

    #[cfg(test)]
    pub mod tests {
        use crate::{
            bench::{City, WeatherMap, decode_lines, parse_number},
            r#gen::Mmap,
        };
        use std::fs::File;
        extern crate test;

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
            let r = m.get(&City::new("aaaaaaaaa".as_bytes())).unwrap();
            assert_eq!(r.count, 2);
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
            let data = Mmap::open::<false>(File::open("./data/measurements.txt").unwrap()).unwrap();
            b.iter(|| {
                let mut m = WeatherMap::default();
                decode_lines(&data, &mut m, false);
            });
        }
    }
}
