//! A minimal reader for numpy `.npz` archives (`numpy.savez`, uncompressed) and the `.npy` files
//! inside them. Written by hand to avoid a zip dependency.
//!
//! `numpy.savez` writes a zip whose entries are STORED (method 0). `savez_compressed` uses
//! DEFLATE, which is rejected with a clear error. Only little-endian `<f4` and `<f8` (converted
//! to `f32`) C-order arrays are accepted, which is all `unified.save_params`
//! (`src/slinky/unified.py:666`) writes.

use crate::error::NetError;
use std::collections::HashMap;

/// One array: its shape and its data in row-major (C) order.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    /// The dimensions.
    pub shape: Vec<usize>,
    /// The elements, row-major.
    pub data: Vec<f32>,
}

fn err<T>(m: impl Into<String>) -> Result<T, NetError> {
    Err(NetError::Npz(m.into()))
}

fn u16_at(b: &[u8], o: usize) -> Result<usize, NetError> {
    match b.get(o..o + 2) {
        Some(s) => Ok(u16::from_le_bytes([s[0], s[1]]) as usize),
        None => err("truncated zip"),
    }
}

fn u32_at(b: &[u8], o: usize) -> Result<usize, NetError> {
    match b.get(o..o + 4) {
        Some(s) => Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize),
        None => err("truncated zip"),
    }
}

/// Parse an `.npz` file held in memory into `name -> array` (the `.npy` suffix is stripped).
pub fn parse_npz(bytes: &[u8]) -> Result<HashMap<String, Array>, NetError> {
    // End of central directory record: signature 0x06054b50, searched from the end.
    if bytes.len() < 22 {
        return err("file too short to be a zip");
    }
    let lo = bytes.len().saturating_sub(22 + 65535);
    let eocd = (lo..=bytes.len() - 22)
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .map_or_else(|| err("no zip end-of-central-directory record"), Ok)?;
    let count = u16_at(bytes, eocd + 10)?;
    let mut p = u32_at(bytes, eocd + 16)?;
    if count == 0xFFFF || u32_at(bytes, eocd + 16)? == 0xFFFF_FFFF {
        return err("zip64 archives are not supported");
    }
    let mut out = HashMap::new();
    for _ in 0..count {
        if bytes.get(p..p + 4) != Some(&[0x50, 0x4b, 0x01, 0x02]) {
            return err("bad central directory entry");
        }
        let method = u16_at(bytes, p + 10)?;
        let csize = u32_at(bytes, p + 20)?;
        let usize_ = u32_at(bytes, p + 24)?;
        let nlen = u16_at(bytes, p + 28)?;
        let xlen = u16_at(bytes, p + 30)?;
        let clen = u16_at(bytes, p + 32)?;
        let local = u32_at(bytes, p + 42)?;
        let name = bytes
            .get(p + 46..p + 46 + nlen)
            .ok_or_else(|| NetError::Npz("truncated zip".into()))?;
        let name = String::from_utf8_lossy(name).into_owned();
        p += 46 + nlen + xlen + clen;
        if csize == 0xFFFF_FFFF || usize_ == 0xFFFF_FFFF {
            return err("zip64 entries are not supported");
        }
        if method != 0 {
            return err(format!(
                "entry {name} is compressed (method {method}); save with numpy.savez, not savez_compressed"
            ));
        }
        if bytes.get(local..local + 4) != Some(&[0x50, 0x4b, 0x03, 0x04]) {
            return err("bad local file header");
        }
        let start = local + 30 + u16_at(bytes, local + 26)? + u16_at(bytes, local + 28)?;
        let data = start
            .checked_add(csize)
            .and_then(|end| bytes.get(start..end))
            .ok_or_else(|| NetError::Npz(format!("entry {name} runs past the end of the file")))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        let arr = parse_npy(data).map_err(|e| match e {
            NetError::Npz(m) => NetError::Npz(format!("{key}: {m}")),
            e => e,
        })?;
        out.insert(key, arr);
    }
    Ok(out)
}

/// Parse the bytes of one `.npy` file.
pub fn parse_npy(b: &[u8]) -> Result<Array, NetError> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return err("not an .npy file");
    }
    let (hlen, hstart) = match b[6] {
        1 => (u16_at(b, 8)?, 10),
        2 | 3 => (u32_at(b, 8)?, 12),
        v => return err(format!("unsupported .npy version {v}")),
    };
    let header = b
        .get(hstart..hstart + hlen)
        .ok_or_else(|| NetError::Npz("truncated .npy header".into()))?;
    let header =
        std::str::from_utf8(header).map_err(|_| NetError::Npz("non-utf8 header".into()))?;
    let descr = dict_value(header, "descr").ok_or_else(|| NetError::Npz("no descr".into()))?;
    let descr = descr.trim().trim_matches('\'').trim_matches('"');
    let fortran = dict_value(header, "fortran_order")
        .ok_or_else(|| NetError::Npz("no fortran_order".into()))?;
    if fortran.trim() != "False" {
        return err("fortran-order arrays are not supported");
    }
    let shape = dict_value(header, "shape").ok_or_else(|| NetError::Npz("no shape".into()))?;
    let shape: Vec<usize> = shape
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| NetError::Npz(format!("bad shape {s:?}")))
        })
        .collect::<Result<_, _>>()?;
    let n = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| NetError::Npz("shape overflow".into()))?;
    let body = &b[hstart + hlen..];
    let (width, wide) = match descr {
        "<f4" | "=f4" => (4, false),
        "<f8" | "=f8" => (8, true),
        d => {
            return err(format!(
                "unsupported dtype {d:?} (need little-endian float32)"
            ));
        }
    };
    let need = n
        .checked_mul(width)
        .ok_or_else(|| NetError::Npz("shape overflow".into()))?;
    if body.len() < need {
        return err(format!(
            "data has {} bytes, shape {shape:?} needs {need}",
            body.len()
        ));
    }
    let data = if wide {
        body[..need]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect()
    } else {
        body[..need]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    Ok(Array { shape, data })
}

/// The text of `key` in a numpy header dict such as `{'descr': '<f4', 'shape': (3, 4), ...}`.
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("'{key}':");
    let i = header.find(&pat)? + pat.len();
    let rest = header[i..].trim_start();
    if rest.starts_with('(') {
        let end = rest.find(')')?;
        Some(&rest[..=end])
    } else {
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(&rest[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny stored zip with one entry, by hand.
    fn tiny_npz(name: &str, npy: &[u8]) -> Vec<u8> {
        let mut z = Vec::new();
        let fname = format!("{name}.npy");
        z.extend_from_slice(&[0x50, 0x4b, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        z.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        z.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        z.extend_from_slice(&[0, 0]);
        z.extend_from_slice(fname.as_bytes());
        z.extend_from_slice(npy);
        let cd = z.len();
        z.extend_from_slice(&[
            0x50, 0x4b, 1, 2, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        z.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        z.extend_from_slice(&(npy.len() as u32).to_le_bytes());
        z.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        z.extend_from_slice(&[0; 12]);
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(fname.as_bytes());
        let cd_len = z.len() - cd;
        z.extend_from_slice(&[0x50, 0x4b, 5, 6, 0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend_from_slice(&(cd_len as u32).to_le_bytes());
        z.extend_from_slice(&(cd as u32).to_le_bytes());
        z.extend_from_slice(&[0, 0]);
        z
    }

    fn npy(shape: &str, vals: &[f32]) -> Vec<u8> {
        let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut v = b"\x93NUMPY\x01\x00".to_vec();
        v.extend_from_slice(&(h.len() as u16).to_le_bytes());
        v.extend_from_slice(h.as_bytes());
        for x in vals {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    #[test]
    fn roundtrip() {
        let z = tiny_npz("a/w", &npy("(2, 3)", &[1., 2., 3., 4., 5., 6.]));
        let m = parse_npz(&z).unwrap();
        assert_eq!(m["a/w"].shape, vec![2, 3]);
        assert_eq!(m["a/w"].data[5], 6.0);
        let z = tiny_npz("b", &npy("(4,)", &[1., 2., 3., 4.]));
        assert_eq!(parse_npz(&z).unwrap()["b"].shape, vec![4]);
    }

    #[test]
    fn bad_input_is_an_error() {
        assert!(parse_npz(&[]).is_err());
        assert!(parse_npz(&[0u8; 100]).is_err());
        let z = tiny_npz("a", &npy("(2, 3)", &[1., 2.]));
        assert!(parse_npz(&z).is_err());
        let mut z = tiny_npz("a", &npy("(1,)", &[1.]));
        let n = z.len();
        z.truncate(n - 5);
        assert!(parse_npz(&z).is_err());
        assert!(parse_npy(b"\x93NUMPY\x01\x00\xff\xff").is_err());
    }
}
