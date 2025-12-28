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
    ($d:expr, $i:expr) => {
        unsafe {
            let r = $i;
            slice::from_raw_parts($d.as_ptr().add(r.start), r.end - r.start)
        }
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
const RESULT_BITS: u32 = 17;

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
                for _ in 0..argv.size {
                    let city = &mut cities[rand!(len)];
                    let temp = format!(
                        ";{}{:.1}\n",
                        if rand!(100) < 50 { "-" } else { "" },
                        rand!(1000) as f32 / 10f32,
                    );
                    $e.write_all(&city.0)?;
                    $e.write_all(temp.as_bytes())?;
                    bar.inc(1);
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
                bench::reduce(
                    &data,
                    argv.slice,
                    argv.workers,
                    argv.dry_run,
                    argv.slice.is_none(),
                )?
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
    use core::{slice, str};
    use proc_cpuinfo::CpuInfo;
    use rapidhash::fast::RandomState;
    use rayon::prelude::*;
    use std::{
        cmp,
        collections::BTreeMap,
        env::args,
        fmt::{self, Display, Formatter},
        fs::File,
        hash::{BuildHasher, Hash, Hasher},
        io::Write,
        iter::FusedIterator,
        mem::ManuallyDrop,
        ops::{AddAssign, Deref},
        ptr,
        simd::{u8x8, u16x8, u32x8, u64x4, u64x8},
        thread,
        time::SystemTime,
    };

    #[derive(Clone, Copy)]
    pub struct Weather {
        min: i16,
        max: i16,
        sum: i64,
        count: u32,
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
    }
    impl From<i16> for Weather {
        fn from(value: i16) -> Self {
            Self::new(value)
        }
    }
    impl AddAssign for Weather {
        fn add_assign(&mut self, rhs: Self) {
            self.sum += rhs.sum;
            self.count += rhs.count;
            self.max = cmp::max(self.max, rhs.max);
            self.min = cmp::min(self.min, rhs.min);
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
                    read_unaligned!(a.as_ptr(), u64).hash(state);
                    read_unaligned!(a.as_ptr().add(a.len() - size_of::<u64>()), u64).hash(state);
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
                use concat_idents::concat_idents;
                macro_rules! cast_x8 {
                    ($ty:expr, $e: expr) => {
                        concat_idents!(ty = u, $ty {
                            unsafe { slice::from_raw_parts::<ty>($e.as_ptr().cast(), 8) }
                        })
                    };
                }
                macro_rules! simd_ne {
                    ($w:expr, $a:expr, $b:expr) => {
                        concat_idents!(smid = u, $w, x8 {
                            <smid>::from_slice(cast_x8!($w, $a))
                                != <smid>::from_slice(cast_x8!($w, $b))
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
                fast_simd(mid_slice!(a, 1..len - 2), mid_slice!(b, 1..len - 2))
            }
        }
    }
    impl<'a> Display for City<'a> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.write_str(unsafe { str::from_utf8_unchecked(self.name) })
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
            if decimal > 0
                && let Some(v) = v.next()
            {
                r.push('.');
                let len = decimal.min(v.len());
                r.push_str(&v[..len]);
                (len..decimal).for_each(|_| r.push('0'));
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
                    match find_newline(pst_slice!(self.data, expect_end)) {
                        Some(i) => expect_end + i + 1,
                        None => self_end,
                    }
                } else {
                    self_end
                };
                self.off = range.end;
                Some(mid_slice!(self.data, range))
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
        pub fn put<const MASK: usize>(&mut self, key: City<'a>, value: i16) -> usize {
            let index = self.hasher.hash_one(key);
            let mut index = ((index.rotate_right(RESULT_BITS * 4)
                ^ (index >> (RESULT_BITS * 3))
                ^ (index >> (RESULT_BITS * 2))
                ^ (index >> RESULT_BITS)
                ^ index)
                & MASK as u64) as usize;
            let mut miss: usize = 0;
            loop {
                match unsafe { self.inner.get_unchecked_mut(index) } {
                    MyWeatherNode::Value((city, weather)) => {
                        if !key.eq(city) {
                            index = (index + 97) & MASK;
                            miss += 1;
                            assert!(miss <= BITS_MASK, "Map is full!");
                            continue;
                        }
                        weather.count += 1;
                        weather.sum += value as i64;
                        match value {
                            x if x > weather.max => weather.max = x,
                            x if x < weather.min => weather.min = x,
                            _ => {}
                        }
                    }
                    node => {
                        *node = MyWeatherNode::Value((key, Weather::new(value)));
                    }
                }
                break miss;
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
            while self.pos < self.inner.len() {
                if let MyWeatherNode::Value(x) = unsafe { self.inner.get_unchecked_mut(self.pos) } {
                    self.pos += 1;
                    return Some(*x);
                }
                self.pos += 1;
            }
            None
        }
    }
    macro_rules! get {
        ($v:expr, $i:expr) => {
            unsafe { *$v.get_unchecked($i) }
        };
    }
    const CHR_NL: u8 = b'\n';
    const CHR_CM: u8 = b';';
    #[inline(always)]
    pub fn find_comma_simd(data: &[u8]) -> Option<usize> {
        const U_SIZE: usize = size_of::<u64>();
        const F_SIZE: usize = size_of::<u64x4>();
        const MASK: u64x4 = u64x4::splat(u64::from_ne_bytes([CHR_CM; U_SIZE]));
        const MASK1: u64x4 = u64x4::splat(u64::from_ne_bytes([0x01; U_SIZE]));
        const MASK2: u64x4 = u64x4::splat(u64::from_ne_bytes([0x80; U_SIZE]));
        let (mut off, end) = (0, data.len());
        // boost performance with simd
        for _ in 0..(end / F_SIZE) {
            let value = read_unaligned!(data.as_ptr().add(off), u64x4) ^ MASK;
            let value = ((value - MASK1) & !value & MASK2).to_array();
            for j in 0..value.len() {
                match get!(value, j) {
                    x if x != 0 => {
                        return Some(off + j * U_SIZE + (x.trailing_zeros() >> 3) as usize);
                    }
                    _ => {}
                }
            }
            off += F_SIZE
        }
        for j in off..end {
            if read_byte!(data.as_ptr(), j) == CHR_CM {
                return Some(j);
            }
        }
        None
    }
    #[inline(always)]
    pub fn find_newline(data: &[u8]) -> Option<usize> {
        const F_SIZE: usize = size_of::<u64>();
        const MASK: u64 = u64::from_ne_bytes([CHR_NL; F_SIZE]);
        const MASK1: u64 = u64::from_ne_bytes([0x01; F_SIZE]);
        const MASK2: u64 = u64::from_ne_bytes([0x80; F_SIZE]);
        let (mut off, end) = (0, data.len());
        // boost performance with swar
        for _ in 0..(end / F_SIZE) {
            let mut value = read_unaligned!(data.as_ptr().add(off), u64) ^ MASK;
            value = (value - MASK1) & !value & MASK2;
            if value != 0 {
                return Some(off + (value.trailing_zeros() >> 3) as usize);
            }
            off += F_SIZE;
        }
        for j in off..end {
            if read_byte!(data.as_ptr(), j) == CHR_NL {
                return Some(j);
            }
        }
        None
    }

    #[repr(transparent)]
    pub struct Reduce<'a>((BTreeMap<City<'a>, Weather>, usize));

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
                    buf.extend_from_slice(format!("{city}={weather}").as_bytes());
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
        debug: bool,
    ) -> Result<Reduce<'_>> {
        fn map<'a>(
            dry_run: bool,
            debug: bool,
        ) -> impl Fn(&'a [u8]) -> (BTreeMap<City<'a>, Weather>, usize) {
            move |part| {
                let clock = SystemTime::now();
                let mut cities = WeatherMap::default();
                let total = decode_lines(part, &mut cities, dry_run, debug);
                if debug {
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
            mut result: (BTreeMap<City<'a>, Weather>, usize),
            cities: (BTreeMap<City<'a>, Weather>, usize),
        ) -> (BTreeMap<City<'a>, Weather>, usize) {
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
                    workers.unwrap_or(cpu.cpu_cores().unwrap()).max(1),
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
                1 => scanner.map(map(dry_run, debug)).reduce(reduce),
                _ => {
                    rayon::ThreadPoolBuilder::new()
                        .thread_name(|i| format!("decode-worker-{i}"))
                        .num_threads(cpu_cores)
                        .build_global()?;
                    scanner
                        .par_bridge()
                        .into_par_iter()
                        .map(map(dry_run, debug))
                        .reduce_with(reduce)
                }
            }
            .unwrap(),
        ))
    }

    pub fn decode_lines<'a>(
        mut data: &'a [u8],
        result: &mut WeatherMap<'a>,
        dry_run: bool,
        debug: bool,
    ) -> usize {
        let mut miss = 0;
        let mut total: usize = 0;
        while let Some(comma) = find_comma_simd(data) {
            let city = pre_slice!(data, comma);
            data = pst_slice!(data, comma + 1);
            match find_newline(data) {
                Some(newline) => {
                    let value = pre_slice!(data, newline);
                    data = pst_slice!(data, newline + 1);
                    if !dry_run {
                        miss += result.put::<BITS_MASK>(city.into(), parse_number(value));
                    }
                    total += 1;
                }
                None => break,
            }
        }
        if debug && !args().any(|a| a == "--bench") {
            eprintln!(
                "Miss Ratio -> {}%",
                (miss as f64 * 1_00f64 / total as f64).format(3)
            );
        }
        total
    }

    #[inline(always)]
    fn parse_number(value: &[u8]) -> i16 {
        macro_rules! m1 {
            ($e:expr) => {
                $e * 10
            };
        }
        macro_rules! m2 {
            ($e:expr) => {
                $e * 100
            };
        }
        // signal bit 001(0)1101 => `-`
        let s = ((!read_byte!(value.as_ptr()) & 0x10) >> 4) as usize;
        // boost performance with swar
        let v = u32::from_be(read_unaligned!(value.as_ptr().add(s), u32))
            >> ((s + 4 - value.len()) << 3);
        (m2!((v >> 24) & 0x0F) + m1!((v >> 16) & 0x0F) + (v & 0x0F)) as i16 * (1 - (s << 1) as i16)
    }

    #[cfg(test)]
    pub mod tests {
        use crate::bench::{City, WeatherMap, decode_lines, parse_number};
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
            decode_lines(data, &mut m, false, false);
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
        fn bench_reduce(b: &mut test::Bencher) {
            let data = "aaaaaaaaa;-10.0\naaaaaaaaa;26.0\ndef;2.1\n".as_bytes();
            let mut m = WeatherMap::default();
            b.iter(|| {
                decode_lines(data, &mut m, false, false);
            });
        }
    }
}
