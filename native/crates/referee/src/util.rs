//! Small helpers: a stable hash for identities and seeds, JSONL files, and times.

use std::fs::OpenOptions;
use std::hash::Hasher;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::Value;

/// FNV-1a, 64 bits: stable across builds and platforms (unlike `DefaultHasher`).
#[derive(Debug, Clone, Copy)]
pub struct Fnv(u64);

impl Fnv {
    /// The FNV offset basis.
    pub fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv::new()
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// 16 hex digits of the FNV-1a of a string.
pub fn hash_hex(text: &str) -> String {
    let mut h = Fnv::new();
    h.write(text.as_bytes());
    format!("{:016x}", h.finish())
}

/// SplitMix64's finaliser: a well-mixed 64-bit function of `x`.
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A seed derived from a list of values (order matters).
pub fn derive(parts: &[u64]) -> u64 {
    parts
        .iter()
        .fold(0x5EED_5EED_5EED_5EEDu64, |acc, &p| mix64(acc ^ mix64(p)))
}

/// The JSON objects of a JSONL file; unreadable (torn or foreign) lines are skipped. A missing
/// file has none.
pub fn read_jsonl(path: &Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(Value::is_object)
        .collect()
}

/// Append one JSON line (one `write`), first ending a line torn by an earlier kill.
pub fn append_jsonl(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    let len = f.metadata()?.len();
    let mut line = String::new();
    if len > 0 {
        let mut last = [0u8; 1];
        f.seek(SeekFrom::Start(len - 1))?;
        f.read_exact(&mut last)?;
        if last[0] != b'\n' {
            line.push('\n');
        }
    }
    line += &value.to_string();
    line.push('\n');
    f.write_all(line.as_bytes())?;
    f.flush()
}

/// The current UTC time as `YYYY-MM-DDTHH:MM:SS+00:00`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// A duration in seconds as `1h02m`, `3m05s` or `4.2s`.
pub fn fmt_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "?".into();
    }
    if seconds >= 3600.0 {
        let m = (seconds / 60.0).round() as u64;
        format!("{}h{:02}m", m / 60, m % 60)
    } else if seconds >= 60.0 {
        let s = seconds.round() as u64;
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{seconds:.1}s")
    }
}

/// A hash of the running executable (the code fingerprint of a record), or `"unknown"`.
pub fn exe_fingerprint() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map_or_else(
            || "unknown".into(),
            |bytes| {
                let mut h = Fnv::new();
                h.write(&bytes);
                format!("{:016x}", h.finish())
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_stable() {
        // FNV-1a test vector.
        let mut h = Fnv::new();
        h.write(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);
        assert_ne!(derive(&[1, 2]), derive(&[2, 1]));
        assert_eq!(derive(&[1, 2]), derive(&[1, 2]));
    }

    #[test]
    fn jsonl_skips_torn_lines_and_repairs_them() {
        let dir = std::env::temp_dir().join(format!("naz-referee-util-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("x.jsonl");
        let _ = std::fs::remove_file(&p);
        assert!(read_jsonl(&p).is_empty());
        append_jsonl(&p, &serde_json::json!({"a": 1})).unwrap();
        // A torn line from a kill mid-write.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{\"a\": 2, \"b")
            .unwrap();
        append_jsonl(&p, &serde_json::json!({"a": 3})).unwrap();
        let v = read_jsonl(&p);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1]["a"], 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn formats() {
        assert_eq!(fmt_duration(4.21), "4.2s");
        assert_eq!(fmt_duration(185.0), "3m05s");
        assert_eq!(fmt_duration(3720.0), "1h02m");
        let now = utc_now();
        assert_eq!(now.len(), 25);
        assert!(now.starts_with("20"));
    }
}
