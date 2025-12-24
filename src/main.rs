#![feature(portable_simd)]
#![feature(test)]
use crate::r#gen::Mmap;
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

    /// slice size, default to file size / workers
    #[arg(short, long)]
    slice: Option<usize>,

    /// parallel workers, default to cpu cores
    #[arg(short, long)]
    workers: Option<usize>,
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
    let mut data = {
        let mut file = File::open(argv.template)?;
        file.read_to_string(&mut data)?;
        data.as_bytes()
    };
    let mut cities: Vec<(bench::City<'_>, bool)> = {
        let mut cities = HashSet::new();
        while let Some(newline) = bench::find_newline(data) {
            if data[0] != b'#'
                && let Some(comma) = bench::find_comma(&data[..newline])
            {
                cities.insert(bench::City::new(&data[..comma]));
            }
            data = &data[newline + 1..];
        }
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
                        "{msg} {wide_bar:.green/blue} {pos:>9}/{len:9} [{elapsed_precise}]",
                    )
                    .unwrap(),
                );
                let len = cities.len();
                for _ in 0..argv.size {
                    let city = &mut cities[r!(len)];
                    let temp = format!(
                        ";{}{}.{}\n",
                        if r!(100) < 50 { "-" } else { "" },
                        r!(100),
                        r!(10),
                    );
                    $e.write_all(&city.0)?;
                    $e.write_all(temp.as_bytes())?;
                    bar.inc(1);
                    city.1 = true
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

    eprintln!(
        "Final size {:.2} {} with {} cities",
        (len * 100f64).round() / 100f64,
        units[i],
        cities.iter().filter(|f| f.1).count()
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
                bench::reduce(&data, argv.slice, argv.workers)?
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
    use anyhow::Result;
    use core::{slice, str};
    use proc_cpuinfo::CpuInfo;
    use rapidhash::{HashMapExt, RapidHashMap};
    use std::{
        cmp,
        collections::BTreeMap,
        fmt::{self, Display, Formatter},
        fs::File,
        hash::{Hash, Hasher},
        io::Write,
        mem::ManuallyDrop,
        ops::{AddAssign, Deref},
        simd::{u8x8, u16x8, u32x8, u64x8},
        sync::{Arc, Mutex, mpsc::channel},
        thread::{self, JoinHandle},
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

    macro_rules! read_byte {
        ($e:expr) => {
            unsafe { *$e }
        };
        ($e:expr, $i:expr) => {
            unsafe { *$e.add($i) }
        };
    }
    macro_rules! read_unaligned {
        ($p:expr, $ty:ty) => {
            unsafe { std::ptr::read_unaligned($p as *const $ty) }
        };
        ($p:expr, $i:expr, $ty:ty) => {
            unsafe { std::ptr::read_unaligned(($p as *const $ty).add($i)) }
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
    impl<'a> Hash for City<'a> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // self.name.hash(state);
            let a = self.name;
            if a.len() < 16 {
                a.hash(state);
            } else {
                read_unaligned!(a.as_ptr(), u128).hash(state);
            }
        }
    }
    impl<'a> PartialEq for City<'a> {
        fn eq(&self, other: &Self) -> bool {
            // self.name.eq(other.name)
            #[inline(always)]
            fn slow_loop(a: &[u8], b: &[u8]) -> bool {
                for i in 0..a.len() {
                    if read_byte!(a.as_ptr(), i) != read_byte!(b.as_ptr(), i) {
                        return false;
                    }
                }
                true
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
                    x if x < 8 => slow_loop(pst_slice!(a, x << 3), pst_slice!(b, x << 3)),
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

    struct Scanner<'a> {
        data: &'a [u8],
        off: Mutex<usize>,
        slice: usize,
    }
    impl<'a> Scanner<'a> {
        fn new(data: &'a [u8], slice: usize) -> Self {
            Self {
                data,
                slice,
                off: Mutex::new(0),
            }
        }
        fn next(&self) -> Option<&'a [u8]> {
            let mut lock = self.off.lock().unwrap();
            let self_end = self.data.len();
            let off = *lock;
            if off < self_end {
                let end = off + self.slice;
                let range = off..if end < self_end {
                    match find_newline(pst_slice!(self.data, end)) {
                        Some(i) => end + i + 1,
                        None => self_end,
                    }
                } else {
                    self_end
                };
                *lock = range.end;
                Some(mid_slice!(self.data, range))
            } else {
                None
            }
        }
    }

    pub type WeatherMap<'a> = RapidHashMap<City<'a>, Weather>;

    macro_rules! find_with_const {
        ($ty:ty, $chr: expr) => {
            const F_SIZE: usize = size_of::<$ty>();
            const MASK: $ty = <$ty>::from_ne_bytes([$chr; size_of::<$ty>()]);
            const MASK1: $ty = <$ty>::from_ne_bytes([0x01; F_SIZE]);
            const MASK2: $ty = <$ty>::from_ne_bytes([0x80; F_SIZE]);
        };
    }
    macro_rules! find_with_mask {
        ($ty:ty, $chr: expr) => {
            #[inline(always)]
            fn find_with_mask(data: &[u8]) -> Option<usize> {
                find_with_const!($ty, $chr);
                let end = data.len();
                // boost performance with swar
                for i in 0..(end / F_SIZE) {
                    let mut value = read_unaligned!(data.as_ptr(), i, $ty) ^ MASK;
                    value = (value - MASK1) & !value & MASK2;
                    if value != 0 {
                        return Some(i * F_SIZE + (value.trailing_zeros() >> 3) as usize);
                    }
                }
                for j in (end - end % F_SIZE)..end {
                    if read_byte!(data.as_ptr(), j) == $chr {
                        return Some(j);
                    }
                }
                None
            }
        };
    }

    #[inline(always)]
    pub fn find_comma(data: &[u8]) -> Option<usize> {
        find_with_mask!(u128, b';');
        find_with_mask(data)
    }
    #[inline(always)]
    pub fn find_newline(data: &[u8]) -> Option<usize> {
        find_with_mask!(u64, b'\n');
        find_with_mask(data)
    }

    pub struct Reduce<'a>(BTreeMap<City<'a>, Weather>, Vec<JoinHandle<()>>);
    impl<'a> Reduce<'a> {
        pub fn write(self, mut file: File, mut buf: ManuallyDrop<Vec<u8>>) -> Result<Self> {
            buf.push(b'{');
            for (id, (city, weather)) in self.0.iter().enumerate() {
                if id != 0 {
                    buf.extend_from_slice(", ".as_bytes());
                }
                buf.extend_from_slice(format!("{city}={weather}").as_bytes());
            }
            buf.extend_from_slice("}\n".as_bytes());
            file.write_all(&buf)?;
            Ok(self)
        }
        pub fn wait(self, clock: SystemTime) -> Result<()> {
            eprintln!(
                "Result in {:.3}ms with {} cities",
                clock.elapsed()?.as_millis(),
                self.0.len(),
            );
            // normally, they are already done.
            for t in self.1.into_iter() {
                t.join().unwrap();
            }
            Ok(())
        }
    }

    pub fn reduce(data: &[u8], slice: Option<usize>, workers: Option<usize>) -> Result<Reduce<'_>> {
        let (tasks, rx) = {
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
            let scanner = Arc::new(Scanner::new(
                unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) },
                slice
                    .unwrap_or(data.len() / cpu_cores)
                    .max(cache_size / cpu_cores),
            ));
            let (tx, rx) = channel::<WeatherMap>();
            let mut tasks = vec![];
            for j in 0..cpu_cores {
                let sc1 = scanner.clone();
                let tx1 = tx.clone();
                tasks.push(
                    thread::Builder::new()
                        .name(format!("decode-task-{j}"))
                        .spawn(move || {
                            let mut cities = WeatherMap::new();
                            while let Some(part) = sc1.next() {
                                decode_lines(part, &mut cities);
                            }
                            tx1.send(cities).unwrap();
                        })?,
                );
            }
            (tasks, rx)
        };

        let mut result = BTreeMap::new();
        for _ in 0..tasks.len() {
            let cities = rx.recv()?;
            for (city, value) in cities {
                result
                    .entry(city)
                    .and_modify(|weather| *weather += value)
                    .or_insert(value);
            }
        }

        Ok(Reduce(result, tasks))
    }

    pub fn decode_lines<'a>(mut data: &'a [u8], cities: &mut WeatherMap<'a>) {
        while let Some(comma) = find_comma(data) {
            let city = City::new(pre_slice!(data, comma));
            data = pst_slice!(data, comma + 1);
            if let Some(newline) = find_newline(data) {
                let value = parse_number(pre_slice!(data, newline));
                data = pst_slice!(data, newline + 1);
                cities
                    .entry(city)
                    .and_modify(|weather| {
                        weather.count += 1;
                        weather.sum += value as i64;
                        match value {
                            x if x > weather.max => weather.max = x,
                            x if x < weather.min => weather.min = x,
                            _ => {}
                        }
                    })
                    .or_insert_with(|| value.into());
            }
        }
    }

    #[inline(always)]
    fn parse_number(value: &[u8]) -> i16 {
        #[inline(always)]
        fn bdc2bin(value: &[u8]) -> i16 {
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
            // boost performance with swar
            let v = (u32::from_be(read_unaligned!(value.as_ptr(), u32)) & 0x0F0F0F0F)
                >> ((4 - value.len()) << 3);
            (m2!(v >> 24) + m1!((v << 8) >> 24) + ((v << 24) >> 24)) as i16
        }

        match read_byte!(value.as_ptr()) {
            b'-' => -bdc2bin(pst_slice!(value, 1)),
            _ => bdc2bin(value),
        }
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
            decode_lines(data, &mut m);
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
        fn bench_decode(b: &mut test::Bencher) {
            let data = "aaaaaaaaa;-10.0\naaaaaaaaa;26.0\ndef;2.1\n".as_bytes();
            let mut m = WeatherMap::default();
            b.iter(|| {
                m.clear();
                decode_lines(data, &mut m);
            });
        }
    }
}
