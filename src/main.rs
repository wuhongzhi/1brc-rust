use crate::r#gen::Mmap;
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
    #[arg(short, long, default_value_t = 100_000_000)]
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
    let mut buf = String::with_capacity(500_000);

    let mut pipe_fds = [0i32; 2];
    unsafe { libc::pipe(pipe_fds.as_mut_ptr()) };
    match fork()? {
        Fork::Parent(_child) => {
            unsafe { libc::close(pipe_fds[1]) }; // Close write end
            let mut reader = unsafe { File::from_raw_fd(pipe_fds[0]) };
            reader.read_to_string(&mut buf)?;
            stdout().write_all(buf.as_bytes())?;
        }
        Fork::Child => {
            unsafe { libc::close(pipe_fds[0]) }; // Close read end
            let data = Mmap::open::<false>(File::open(argv.data)?)?;
            let result = bench::reduce(&data, argv.parts)?;
            //format & detail report
            buf.push('{');
            for (id, (city, weather)) in result.iter().enumerate() {
                if id != 0 {
                    buf.push_str(", ");
                }
                buf.push_str(&format!("{city}={weather}"));
            }
            buf.push_str("}\n");
            eprintln!(
                "Result in {:.3}ms with {} cities",
                clock.elapsed()?.as_millis(),
                result.len(),
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
        collections::BTreeMap,
        fmt::{self, Display, Formatter},
        hash::{Hash, Hasher},
        ops::{AddAssign, Deref},
        ptr,
        sync::{Arc, Mutex, mpsc::channel},
        thread,
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
            unsafe { ptr::read_unaligned($p as *const $ty) }
        };
        ($p:expr, $i:expr, $ty:ty) => {
            unsafe { ptr::read_unaligned($p.add($i) as *const $ty) }
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

            let p1 = self.name.as_ptr();
            if self.name.len() >= 16 {
                read_unaligned!(p1, u128).hash(state);
            } else {
                self.name.hash(state);
            }
        }
    }
    impl<'a> PartialEq for City<'a> {
        fn eq(&self, other: &Self) -> bool {
            self.name.eq(other.name)
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

    #[inline(always)]
    fn find_with_mask<const MASK: usize, const CHR: u8>(data: &[u8]) -> Option<usize> {
        let (mut i, end) = (0, data.len());
        //boost performance with swar
        const SIZE: usize = size_of::<usize>();
        const MASK1: usize = usize::from_ne_bytes([0x01; SIZE]);
        const MASK2: usize = usize::from_ne_bytes([0x80; SIZE]);
        while i + SIZE < end {
            let input = read_unaligned!(data.as_ptr(), i, usize) ^ MASK;
            let p = (input - MASK1) & !input & MASK2;
            if p != 0 {
                return Some(i + (p.trailing_zeros() >> 3) as usize);
            }
            i += SIZE;
        }
        while i < end {
            if peek!(data, i) == CHR {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    const CHR1: u8 = b';';
    const CHR2: u8 = b'\n';
    const MASK1: usize = usize::from_ne_bytes([CHR1; size_of::<usize>()]);
    const MASK2: usize = usize::from_ne_bytes([CHR2; size_of::<usize>()]);

    #[inline(always)]
    fn find_comma(data: &[u8]) -> Option<usize> {
        find_with_mask::<MASK1, CHR1>(data)
    }
    #[inline(always)]
    fn find_newline(data: &[u8]) -> Option<usize> {
        find_with_mask::<MASK2, CHR2>(data)
    }

    pub fn reduce(data: &[u8], parts: Option<usize>) -> Result<BTreeMap<City<'_>, Weather>> {
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
                        if !part.is_empty() {
                            tx1.send(Some(decode_lines(part))).unwrap();
                        }
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

        // just wait all, normal they all down here.
        // tasks.into_iter().for_each(|t| t.join().unwrap());

        Ok(result)
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

        while let Some(p) = find_comma(data)
            && p > 0
        {
            let city = &data[..p];
            data = &data[p + 1..];
            match find_newline(data) {
                Some(p) if p > 0 => {
                    compute(&mut batch, city, &data[..p]);
                    data = &data[p + 1..];
                }
                _ => break,
            }
        }
        batch
    }

    #[inline(always)]
    fn parse_number(mut temp: &[u8]) -> i32 {
        let x = match peek!(temp) {
            b'-' => {
                temp = &temp[1..];
                |v: u32| -(v as i32)
            }
            _ => |v: u32| v as i32,
        };

        if temp.len() > 4 {
            temp = &temp[..4]
        }

        macro_rules! m1 {
            ($e:expr) => {
                // (10_u32 * $e)
                ($e << 3) + ($e << 1)
            };
        }
        macro_rules! m2 {
            ($e:expr) => {
                // (100_u32 * $e)
                m1!(m1!($e))
            };
        }

        //boost performance with swar
        let v = match temp.len() {
            3 => u32::from_be(unsafe { ptr::read_unaligned(temp.as_ptr() as *const u32) }) >> 8,
            4 => u32::from_be(unsafe { ptr::read_unaligned(temp.as_ptr() as *const u32) }),
            x => unreachable!("bad number {x}"),
        };
        x(m2!((v >> 24) & 0x0F) + m1!((v >> 16) & 0x0F) + (v & 0x0F))
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
            let m = decode_lines(data);
            assert_eq!(m.len(), 2);
            let r = m.get(&City::new("aaaaaaaa".as_bytes())).unwrap();
            assert_eq!(r.count, 2);
            assert_eq!(r.min, -100);
            assert_eq!(r.max, 260);
            assert_eq!(r.sum, 160);
        }
    }
}
