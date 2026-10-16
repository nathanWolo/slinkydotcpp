//! Python's text forms of request values: `json.dumps`, `str()` and `repr()`.
//!
//! The move log and the log lines show request values (the game id, the turn, the ruleset
//! name, ...) the way the Python server prints them, so a log line from this server matches
//! the Python server's byte for byte:
//!
//! * [`dumps`]: `json.dumps(value)` with the default arguments: `", "` and `": "` separators,
//!   `ensure_ascii` (every character outside `' '..='~'` escaped, astral ones as surrogate
//!   pairs, lower-case hex), `NaN` / `Infinity` / `-Infinity` for non-finite floats, floats in
//!   `repr` form;
//! * [`py_str`] and [`py_repr`]: `str(value)` and `repr(value)` (`None`, `True`, `5`, `5.0`,
//!   `'a'`, `[1, 'a']`, `{'k': None}`);
//! * [`float_repr`]: `repr(float)` (shortest round-trip digits, exponent form below `1e-4`
//!   and from `1e16`, `.0` on integral values).
//!
//! Python raises `RecursionError` on values nested about a thousand levels deep; these
//! functions return [`TooDeep`] beyond [`MAX_NEST`] levels (no deep recursion, no stack
//! overflow). `repr` of a string treats every character from U+00A1 up as printable, which is
//! Python's `str.isprintable` for everything an engine sends (Python also escapes some rare
//! Unicode categories, such as unassigned code points and non-ASCII spaces other than U+00A0).

use slinky_api::PyValue;
use std::fmt::Write;

/// Deepest nesting printed; Python's recursion limit (1000 frames, some already in use).
pub const MAX_NEST: usize = 900;

/// A value nested deeper than [`MAX_NEST`]: Python raises `RecursionError` there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooDeep;

impl std::fmt::Display for TooDeep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecursionError: maximum recursion depth exceeded")
    }
}

/// `repr(x)` of a float.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` prints the shortest digits that round-trip, as Python's repr does.
    let e = format!("{x:e}");
    let (mantissa, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let mut out = String::new();
    if x < 0.0 {
        out.push('-');
    }
    // value = 0.d1d2...dn * 10^decpt
    let decpt = exp + 1;
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exp < 0 { '-' } else { '+' };
        let _ = write!(out, "e{sign}{:02}", exp.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if decpt >= n {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (decpt - n) as usize));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// `repr(round(x, places))` for a float well inside `1e15` (the move log's `t` and `ms`).
pub fn rounded_repr(x: f64, places: usize) -> String {
    // `{:.N}` rounds the exact binary value half to even, as Python's round() does.
    let s = format!("{x:.places$}");
    s.parse::<f64>().map_or(s, float_repr)
}

/// A JSON string literal as `json.dumps` writes it (`ensure_ascii=True`).
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    push_json_string(&mut out, s);
    out
}

fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// `json.dumps(value)` (module docs).
pub fn dumps(value: &PyValue) -> Result<String, TooDeep> {
    let mut out = String::new();
    dump_into(&mut out, value, 0)?;
    Ok(out)
}

fn dump_into(out: &mut String, value: &PyValue, depth: usize) -> Result<(), TooDeep> {
    if depth > MAX_NEST {
        return Err(TooDeep);
    }
    match value {
        PyValue::None => out.push_str("null"),
        PyValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        PyValue::Int(i) => {
            let _ = write!(out, "{i}");
        }
        PyValue::BigInt { negative, digits } => {
            if *negative {
                out.push('-');
            }
            out.push_str(digits);
        }
        PyValue::Float(f) => out.push_str(&json_float(*f)),
        PyValue::Str(s) => push_json_string(out, s),
        PyValue::List(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_into(out, v, depth + 1)?;
            }
            out.push(']');
        }
        PyValue::Dict(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                push_json_string(out, k);
                out.push_str(": ");
                dump_into(out, v, depth + 1)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// A float as `json.dumps` writes it.
pub fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        float_repr(f)
    }
}

/// `str(value)`: a string as itself, anything else as its `repr`.
pub fn py_str(value: &PyValue) -> Result<String, TooDeep> {
    match value {
        PyValue::Str(s) => Ok(s.clone()),
        other => py_repr(other),
    }
}

/// `str(value)` of an optional value (`None` when missing, as `dict.get` gives).
pub fn py_str_opt(value: Option<&PyValue>) -> Result<String, TooDeep> {
    value.map_or_else(|| Ok("None".to_string()), py_str)
}

/// `repr(value)`.
pub fn py_repr(value: &PyValue) -> Result<String, TooDeep> {
    let mut out = String::new();
    repr_into(&mut out, value, 0)?;
    Ok(out)
}

fn repr_into(out: &mut String, value: &PyValue, depth: usize) -> Result<(), TooDeep> {
    if depth > MAX_NEST {
        return Err(TooDeep);
    }
    match value {
        PyValue::None => out.push_str("None"),
        PyValue::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        PyValue::Int(i) => {
            let _ = write!(out, "{i}");
        }
        PyValue::BigInt { negative, digits } => {
            if *negative {
                out.push('-');
            }
            out.push_str(digits);
        }
        PyValue::Float(f) => out.push_str(&float_repr(*f)),
        PyValue::Str(s) => push_str_repr(out, s),
        PyValue::List(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr_into(out, v, depth + 1)?;
            }
            out.push(']');
        }
        PyValue::Dict(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                push_str_repr(out, k);
                out.push_str(": ");
                repr_into(out, v, depth + 1)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// `repr(s)` of a string: single quotes unless it holds `'` and no `"`.
fn push_str_repr(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..=0xa0).contains(&(c as u32)) => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

#[cfg(test)]
mod tests {
    use super::*;
    use slinky_api::py::parse;

    #[test]
    fn floats_print_as_python_repr() {
        for (x, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (46.9, "46.9"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-2.5e300, "-2.5e+300"),
            (5e-324, "5e-324"),
            (1792114362.887, "1792114362.887"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
        assert_eq!(rounded_repr(1792114362.8801, 3), "1792114362.88");
        assert_eq!(rounded_repr(33.0, 1), "33.0");
        assert_eq!(rounded_repr(46.94999, 1), "46.9");
        assert_eq!(rounded_repr(0.04, 1), "0.0");
        assert_eq!(rounded_repr(0.25, 1), "0.2"); // half to even on the exact value
    }

    #[test]
    fn dumps_matches_json_dumps() {
        let cases = [
            (
                r#"{"a": 1, "b": [true, null, 2.5, "x"]}"#,
                r#"{"a": 1, "b": [true, null, 2.5, "x"]}"#,
            ),
            (r#"{}"#, "{}"),
            (r#"[]"#, "[]"),
            (
                r#"[NaN, Infinity, -Infinity, 1e400]"#,
                "[NaN, Infinity, -Infinity, Infinity]",
            ),
            (
                r#""caf\u00e9 \ud83d\ude00 \u007f\n\t\"\\/""#,
                r#""caf\u00e9 \ud83d\ude00 \u007f\n\t\"\\/""#,
            ),
            (
                "123456789012345678901234567890123456789012",
                "123456789012345678901234567890123456789012",
            ),
            ("1E2", "100.0"),
        ];
        for (text, want) in cases {
            let v = parse(text).unwrap();
            assert_eq!(dumps(&v).unwrap(), want, "{text}");
        }
        let deep = format!("{}{}", "[".repeat(2000), "]".repeat(2000));
        assert_eq!(dumps(&parse(&deep).unwrap()), Err(TooDeep));
        assert_eq!(json_string("\u{1}\u{8}\u{c}"), r#""\u0001\b\f""#);
    }

    #[test]
    fn str_and_repr_match_python() {
        let cases = [
            ("null", "None"),
            ("true", "True"),
            ("5", "5"),
            ("5.0", "5.0"),
            (r#""g-1""#, "g-1"),
            (r#"[1, "a", null, false]"#, "[1, 'a', None, False]"),
            (
                r#"{"k": "it's", "j": "say \"hi\" it's"}"#,
                r#"{'k': "it's", 'j': 'say "hi" it\'s'}"#,
            ),
            (r#"["a\nb\u0001\u00a0\u00e9"]"#, "['a\\nb\\x01\\xa0\u{e9}']"),
        ];
        for (text, want) in cases {
            assert_eq!(py_str(&parse(text).unwrap()).unwrap(), want, "{text}");
        }
        assert_eq!(py_str_opt(None).unwrap(), "None");
    }
}
