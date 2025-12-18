#![feature(portable_simd)]
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

    /// how many parts in file, default to number of physical cpu
    #[arg(short, long)]
    parts: Option<usize>,
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
        .into_keys()
        .filter(|x| !x.name.is_empty() && x.name[0] != b'#')
        .take(argv.cities)
        .collect();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&argv.data)?;
    let mut result = HashSet::new();
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
                        "{msg} {wide_bar:.green/blue} {pos:>9}/{len:9} [{elapsed_precise}]",
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
                    result.insert(city);
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
        result.len()
    );
    Ok(())
}

fn bench(argv: BenchArg) -> Result<()> {
    let clock = SystemTime::now();

    let mut pipe_fds = [0i32; 2];
    unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
    match fork()? {
        Fork::Parent(_child) => {
            unsafe { libc::close(pipe_fds[1]) }; // Close write end
            let mut buf = String::with_capacity(512_000);
            let mut reader = unsafe { File::from_raw_fd(pipe_fds[0]) };
            reader.read_to_string(&mut buf)?;
            stdout().write_all(buf.as_bytes())?;
        }
        Fork::Child => {
            unsafe { libc::close(pipe_fds[0]) }; // Close read end
            match Mmap::open::<false>(File::open(argv.data)?) {
                Ok(data) => {
                    bench::reduce(&data, argv.parts)?
                        .write(unsafe { File::from_raw_fd(pipe_fds[1]) })?
                        .wait(clock)?;
                }
                Err(e) => {
                    eprintln!("{e:?}");
                }
            }
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
    use anyhow::{Ok, Result};
    use core::{slice, str};
    use rapidhash::{HashMapExt, RapidHashMap};
    use std::{
        collections::BTreeMap,
        fmt::{self, Display, Formatter},
        fs::File,
        hash::{Hash, Hasher},
        io::Write,
        ops::{AddAssign, Deref},
        simd::{u8x8, u16x8, u32x8, u64x8},
        sync::{Arc, Mutex, mpsc::channel},
        thread::{self, JoinHandle},
        time::SystemTime,
    };

    #[derive(Clone, Copy)]
    pub struct Weather {
        min: i32,
        max: i32,
        sum: i64,
        count: u64,
    }
    impl Weather {
        fn new(value: i32) -> Self {
            Self {
                min: value,
                max: value,
                sum: value as i64,
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

    macro_rules! read_unaligned {
        ($p:expr, $ty:ty) => {
            unsafe { std::ptr::read_unaligned($p as *const $ty) }
        };
        ($p:expr, $i:expr, $ty:ty) => {
            unsafe { std::ptr::read_unaligned($p.add($i) as *const $ty) }
        };
    }

    #[repr(transparent)]
    #[derive(Eq, PartialOrd, Ord, Clone)]
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
                let mut i = 0;
                let len = a.len();
                while i < len && peek!(a, i) == peek!(b, i) {
                    i += 1;
                }
                i == len
            }
            #[inline(always)]
            fn fast_simd(a: &[u8], b: &[u8]) -> bool {
                macro_rules! cast_x8 {
                    ($ty:ty, $e: expr) => {
                        unsafe { slice::from_raw_parts::<$ty>(a.as_ptr().cast(), 8) }
                    };
                }
                macro_rules! simd_ne {
                    ($smid:ty, $ty:ty, $a:expr, $b:expr) => {
                        <$smid>::from_slice(cast_x8!($ty, $a))
                            != <$smid>::from_slice(cast_x8!($ty, $b))
                    };
                    ($smid1:ty, $ty1:ty, $c:expr, $smid2:ty, $ty2:ty, $a:expr, $b:expr) => {
                        simd_ne!($smid1, $ty1, $a, $b)
                            || simd_ne!($smid2, $ty2, &$a[$c..], &$b[$c..])
                    };
                    ($smid1:ty, $ty1:ty, $c:expr, $smid2:ty, $ty2:ty, $d:expr, $smid3:ty, $ty3:ty, $a:expr, $b:expr) => {
                        simd_ne!($smid1, $ty1, $c, $smid2, $ty2, $a, $b)
                            || simd_ne!($smid3, $ty3, &$a[$c + $d..], &$b[$c + $d..])
                    };
                }
                match a.len() >> 3 {
                    1 if simd_ne!(u8x8, u8, a, b) => false,
                    2 if simd_ne!(u16x8, u16, a, b) => false,
                    3 if simd_ne!(u16x8, u16, 16, u8x8, u8, a, b) => false,
                    4 if simd_ne!(u32x8, u32, a, b) => false,
                    5 if simd_ne!(u32x8, u32, 32, u8x8, u8, a, b) => false,
                    6 if simd_ne!(u32x8, u32, 32, u16x8, u16, a, b) => false,
                    7 if simd_ne!(u32x8, u32, 32, u16x8, u16, 16, u8x8, u8, a, b) => false,
                    8 if simd_ne!(u64x8, u64, a, b) => false,
                    x if x < 8 => slow_loop(&a[x << 3..], &b[x << 3..]),
                    _ => a == b,
                }
            }

            #[inline(always)]
            //fast detect first & last 2bytes base on the city statistics
            fn fast_detect(a: &[u8], b: &[u8]) -> bool {
                let len = a.len();
                if peek!(a) != peek!(b)
                    || read_unaligned!(a.as_ptr(), len - 2, u16)
                        != read_unaligned!(b.as_ptr(), len - 2, u16)
                {
                    return false;
                }

                true
            }

            #[inline(always)]
            fn fast_length(a: usize, b: usize) -> bool {
                a != b
            }

            let mut a = self.name;
            let mut b = other.name;
            let mut len = a.len();
            if fast_length(len, b.len()) {
                return false;
            }
            if len >= 3 {
                match fast_detect(a, b) {
                    false => return false,
                    true if len == 3 => return true,
                    true => {
                        len -= 3;
                        a = &a[1..=len];
                        b = &b[1..=len];
                    }
                }
            }
            if len < 8 {
                return slow_loop(a, b);
            }
            fast_simd(a, b)
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
        slice: usize,
        end: usize,
        start: Mutex<usize>,
    }
    impl<'a> Scanner<'a> {
        fn new(data: &'a [u8], parts: usize) -> Self {
            let end = data.len();
            Self {
                end,
                data,
                slice: end / parts,
                start: Mutex::new(0),
            }
        }
        fn next(&self) -> Option<&'a [u8]> {
            let mut lock = self.start.lock().unwrap();
            let start = *lock;
            if start < self.end {
                let end = start + self.slice;
                let range = start..if end >= self.end {
                    self.end
                } else {
                    match find_newline(&self.data[end..]) {
                        Some(i) => end + i + 1,
                        None => self.end,
                    }
                };
                *lock = range.end;
                // eprintln!("{:?}=>{:?}", thread::current().id(), range);
                Some(&self.data[range])
            } else {
                None
            }
        }
    }

    type WeatherHashMap<'a> = RapidHashMap<City<'a>, Weather>;

    const C_CHR: u8 = b';';
    const N_CHR: u8 = b'\n';
    const F_SIZE: usize = size_of::<usize>();
    const FC_MASK: usize = usize::from_ne_bytes([C_CHR; size_of::<usize>()]);
    const FN_MASK: usize = usize::from_ne_bytes([N_CHR; size_of::<usize>()]);
    const F1_MASK: usize = usize::from_ne_bytes([0x01; F_SIZE]);
    const F2_MASK: usize = usize::from_ne_bytes([0x80; F_SIZE]);

    #[inline(always)]
    fn find_with_mask<const MASK: usize, const MASK1: usize, const MASK2: usize, const CHR: u8>(
        data: &[u8],
    ) -> Option<usize> {
        let (mut i, end) = (0, data.len());
        //boost performance with swar
        while i + F_SIZE < end {
            let input = read_unaligned!(data.as_ptr(), i, usize) ^ MASK;
            let position = (input - MASK1) & !input & MASK2;
            if position != 0 {
                return Some(i + (position.trailing_zeros() >> 3) as usize);
            }
            i += F_SIZE;
        }
        while i < end && peek!(data, i) != CHR {
            i += 1;
        }
        if i < end { Some(i) } else { None }
    }

    #[inline(always)]
    fn find_comma(data: &[u8]) -> Option<usize> {
        find_with_mask::<FC_MASK, F1_MASK, F2_MASK, C_CHR>(data)
    }
    #[inline(always)]
    fn find_newline(data: &[u8]) -> Option<usize> {
        find_with_mask::<FN_MASK, F1_MASK, F2_MASK, N_CHR>(data)
    }

    pub struct Reduce<'a>(BTreeMap<City<'a>, Weather>, Vec<JoinHandle<()>>);
    impl<'a> Reduce<'a> {
        pub fn write(self, mut file: File) -> Result<Self> {
            let mut buf = String::with_capacity(512_000);
            buf.push('{');
            for (id, (city, weather)) in self.0.iter().enumerate() {
                if id != 0 {
                    buf.push_str(", ");
                }
                buf.push_str(&format!("{city}={weather}"));
            }
            buf.push_str("}\n");
            file.write_all(buf.as_bytes())?;
            Ok(self)
        }
        pub fn wait(self, clock: SystemTime) -> Result<()> {
            eprintln!(
                "Result in {:.3}ms with {} cities",
                clock.elapsed()?.as_millis(),
                self.0.len(),
            );
            // just wait all, normal they all down here.
            for t in self.1.into_iter() {
                t.join().unwrap();
            }
            Ok(())
        }
    }

    pub fn reduce(data: &[u8], parts: Option<usize>) -> Result<Reduce<'_>> {
        let (tasks, rx) = {
            let (tx, rx) = channel::<Option<WeatherHashMap>>();
            let jobs = num_cpus::get_physical();
            let scanner = Arc::new(Scanner::new(
                unsafe { slice::from_raw_parts(data.as_ptr(), data.len()) },
                parts.unwrap_or(0).max(jobs),
            ));

            let mut tasks = vec![];
            for _ in 0..jobs {
                let sc1 = scanner.clone();
                let tx1 = tx.clone();
                tasks.push(thread::spawn(move || {
                    while let Some(part) = sc1.next() {
                        tx1.send(Some(decode_lines(part))).unwrap();
                    }
                    tx1.send(None).unwrap();
                }));
            }
            (tasks, rx)
        };

        let mut finish = 0;
        let mut result = BTreeMap::new();
        while finish < tasks.len() {
            match rx.recv()? {
                Some(batch) => {
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

        Ok(Reduce(result, tasks))
    }

    pub fn decode_lines<'a>(mut data: &'a [u8]) -> WeatherHashMap<'a> {
        let mut batch = WeatherHashMap::with_capacity(10_000);
        #[inline(always)]
        fn compute<'a>(batch: &mut WeatherHashMap<'a>, city: &'a [u8], temp: &'a [u8]) {
            batch
                .entry(City::new(city))
                .and_modify(|w| {
                    let value = parse_number(temp);
                    w.count += 1;
                    w.sum += value as i64;
                    if value > w.max {
                        w.max = value
                    } else if value < w.min {
                        w.min = value;
                    }
                })
                .or_insert_with(|| Weather::new(parse_number(temp)));
        }

        while let Some(comma) = find_comma(data)
            && comma > 0
        {
            let city = &data[..comma];
            data = &data[comma + 1..];
            match find_newline(data) {
                Some(newline) if newline > 0 => {
                    compute(&mut batch, city, &data[..newline]);
                    data = &data[newline + 1..];
                }
                _ => break,
            }
        }
        batch
    }

    #[inline(always)]
    fn parse_number(bcd: &[u8]) -> i32 {
        #[inline(always)]
        fn bdc2bin(bcd: &[u8]) -> i32 {
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
            //boost performance with swar
            let v = (u32::from_be(read_unaligned!(bcd.as_ptr(), u32)) & 0x0F0F0F0F)
                >> ((4 - bcd.len()) << 3);
            (m2!(v >> 24) + m1!((v << 8) >> 24) + ((v << 24) >> 24)) as i32
        }

        match peek!(bcd) {
            b'-' => -bdc2bin(&bcd[1..]),
            _ => bdc2bin(bcd),
        }
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
            let data = "aaaaaaaaa;-10.0\naaaaaaaaa;26.0\ndef;2.1\n".as_bytes();
            let m = decode_lines(data);
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
            //boost performance with swar
            const SIZE: usize = size_of::<u32>();
            const MASK: u32 = u32::from_ne_bytes([b';'; SIZE]);
            const MASK1: u32 = u32::from_ne_bytes([0x01; SIZE]);
            const MASK2: u32 = u32::from_ne_bytes([0x80; SIZE]);
            let mut arr = [b'1'; SIZE];
            arr[SIZE - 2] = b';';
            let mut input = u32::from_ne_bytes(arr);
            let mut buf = String::new();

            e!(buf, input, "");
            e!(buf, MASK, "");
            e!(buf, input ^ MASK, "=>input");
            input ^= MASK;
            e!(buf, MASK1, "");
            e!(buf, (input - MASK1), "");
            e!(buf, !input, "");
            e!(buf, (input - MASK1) & !input, "");
            e!(buf, MASK2, "");
            e!(buf, (input - MASK1) & !input & MASK2, "");
            let p = (input - MASK1) & !input & MASK2;
            e!(buf, p.trailing_zeros());
            e!(buf, p.trailing_zeros() >> 3);
            assert_eq!(p.trailing_zeros() >> 3, (SIZE - 2) as u32);

            eprintln!("\n\n{}\n", buf.as_str());
        }
    }
}
