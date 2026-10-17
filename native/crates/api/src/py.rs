//! A request body as the Python server sees it: `json.loads` on the body bytes
//! (server.py:657), then dynamically typed reads.
//!
//! The Python server never validates a request against a schema. It reads the fields it needs
//! with `req["board"]`, `len(...)`, comparisons and numpy stores, so what it accepts is defined
//! by Python's object semantics: `1 == 1.0 == True`, a float health is truncated by numpy, a
//! numeric string health is parsed, `None` read as a number raises, a dict iterates over its
//! keys, and so on. This module gives the conversion (`convert.rs`) and the fallback move
//! (`safe.rs`) the same semantics:
//!
//! * [`PyValue`]: a parsed JSON value with Python's types (arbitrary-size integers keep their
//!   decimal digits, floats may be NaN or infinite, objects keep the last of duplicate keys);
//! * [`parse`]: Python's `json` dialect (`json.decoder`, C scanner, `strict=True`): `NaN`,
//!   `Infinity` and `-Infinity` literals, duplicate keys (the last wins, at the first key's
//!   position), integers of up to 4300 digits (`sys.int_info.default_max_str_digits`), nesting
//!   up to the C recursion limit (9997 containers on Python 3.12, measured), lone UTF-16
//!   surrogates (kept by Python, read as U+FFFD here: Rust strings cannot hold them). It never
//!   recurses, so no input can overflow the stack;
//! * Python operations that may raise ([`PyErr`]): subscripting (`item`), `dict.get`, `len`,
//!   iteration, `==`, `<`, `+`, `*`, `%`, `int()` ([`python_int`]) and numpy's int32 store
//!   ([`int32`]), with Python's cross-type rules.
//!
//! Integers beyond `i128` (about 1.7e38) keep their digits: comparisons, equality, addition
//! and the modulo by a smaller integer are exact. Only a product beyond `i128` (kept by sign,
//! for `check_sizes`) and the modulo of two such integers (an error) are not.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// A JSON value with Python's types, as `json.loads` returns it.
///
/// Dropping a deeply nested value does not recurse (see the `Drop` impl); the derived `Debug`
/// and `PartialEq` do, so use them on small values only (tests).
#[derive(Debug, PartialEq)]
pub enum PyValue {
    /// `null` (`None`).
    None,
    /// `true` / `false`. Python's `bool` is an `int` (`True == 1`, `True + 1 == 2`).
    Bool(bool),
    /// An integer that fits in `i128`.
    Int(i128),
    /// An integer beyond `i128`: its sign and decimal digits (no leading zeros).
    BigInt { negative: bool, digits: String },
    /// A float, possibly NaN or infinite (`NaN`, `Infinity`, `1e400`).
    Float(f64),
    /// A string.
    Str(String),
    /// An array (`list`).
    List(Vec<PyValue>),
    /// An object (`dict`): keys in first-insertion order, each with its last value.
    Dict(Vec<(String, PyValue)>),
}

impl Drop for PyValue {
    /// Drops the children through an explicit stack: a body may nest about ten thousand
    /// containers, which the default recursive drop would turn into a deep call stack.
    fn drop(&mut self) {
        let mut stack: Vec<PyValue> = Vec::new();
        take_children(self, &mut stack);
        while let Some(mut v) = stack.pop() {
            take_children(&mut v, &mut stack);
            // `v` now has no children and drops without recursion.
        }
    }
}

fn take_children(v: &mut PyValue, out: &mut Vec<PyValue>) {
    match v {
        PyValue::List(items) => out.append(items),
        PyValue::Dict(items) => out.extend(std::mem::take(items).into_iter().map(|(_, v)| v)),
        _ => {}
    }
}

/// A Python exception (`TypeError`, `KeyError`, `ValueError`, `OverflowError`, ...): what the
/// server would have raised, as a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyErr(pub String);

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The result of a Python operation.
pub type PyResult<T> = Result<T, PyErr>;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyErr(msg.into()))
}

impl PyValue {
    /// Python's type name, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            PyValue::None => "NoneType",
            PyValue::Bool(_) => "bool",
            PyValue::Int(_) | PyValue::BigInt { .. } => "int",
            PyValue::Float(_) => "float",
            PyValue::Str(_) => "str",
            PyValue::List(_) => "list",
            PyValue::Dict(_) => "dict",
        }
    }

    /// `self.get(key)` of a dict; `None` for a missing key or a value that is not a dict.
    pub fn get(&self, key: &str) -> Option<&PyValue> {
        match self {
            PyValue::Dict(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Whether this is a dict.
    pub fn is_dict(&self) -> bool {
        matches!(self, PyValue::Dict(_))
    }

    /// Python truthiness (`bool(x)`).
    pub fn truthy(&self) -> bool {
        match self {
            PyValue::None => false,
            PyValue::Bool(b) => *b,
            PyValue::Int(i) => *i != 0,
            PyValue::BigInt { .. } => true,
            PyValue::Float(f) => *f != 0.0,
            PyValue::Str(s) => !s.is_empty(),
            PyValue::List(a) => !a.is_empty(),
            PyValue::Dict(d) => !d.is_empty(),
        }
    }

    /// `self[key]` with a string key: a dict's value, else `KeyError` / `TypeError`.
    pub fn item(&self, key: &str) -> PyResult<&PyValue> {
        match self {
            PyValue::Dict(_) => self
                .get(key)
                .map_or_else(|| err(format!("KeyError: {key:?}")), Ok),
            other => err(format!(
                "TypeError: {} indices are not str ({key:?})",
                other.type_name()
            )),
        }
    }

    /// `self.get(key)` called as a method: `AttributeError` unless this is a dict.
    pub fn dict_get(&self, key: &str) -> PyResult<Option<&PyValue>> {
        match self {
            PyValue::Dict(_) => Ok(self.get(key)),
            other => err(format!(
                "AttributeError: '{}' object has no attribute 'get'",
                other.type_name()
            )),
        }
    }

    /// `len(self)`: a str counts code points, a list its items, a dict its keys.
    pub fn len(&self) -> PyResult<usize> {
        match self {
            PyValue::Str(s) => Ok(s.chars().count()),
            PyValue::List(a) => Ok(a.len()),
            PyValue::Dict(d) => Ok(d.len()),
            other => err(format!(
                "TypeError: object of type '{}' has no len()",
                other.type_name()
            )),
        }
    }

    /// Whether `len(self) == 0` (an error when `len` is).
    pub fn is_empty(&self) -> PyResult<bool> {
        Ok(self.len()? == 0)
    }

    /// The items of `for x in self` when every `x` is then subscripted with a string
    /// (`x["id"]`, `p["x"]`): a list's items. A str iterates over one-character strings and a
    /// dict over its keys, which such a subscript refuses, so they are errors unless empty
    /// (no iteration at all); anything else is not iterable.
    pub fn subscripted_items(&self) -> PyResult<&[PyValue]> {
        match self {
            PyValue::List(a) => Ok(a),
            PyValue::Str(s) if s.is_empty() => Ok(&[]),
            PyValue::Dict(d) if d.is_empty() => Ok(&[]),
            PyValue::Str(_) | PyValue::Dict(_) => err("TypeError: string indices must be integers"),
            other => err(format!(
                "TypeError: '{}' object is not iterable",
                other.type_name()
            )),
        }
    }

    /// The number this value is in arithmetic and comparisons (`bool` is an `int`).
    pub fn num(&self) -> Option<Num> {
        match self {
            PyValue::Bool(b) => Some(Num::Int(i128::from(*b))),
            PyValue::Int(i) => Some(Num::Int(*i)),
            PyValue::BigInt { negative, digits } => Some(Num::Big {
                negative: *negative,
                digits: digits.clone(),
            }),
            PyValue::Float(f) => Some(Num::Float(*f)),
            _ => None,
        }
    }
}

// --- Numbers -----------------------------------------------------------------------------------

/// A Python number: `int` (in `i128`, or beyond it with its digits) or `float`.
#[derive(Debug, Clone, PartialEq)]
pub enum Num {
    Int(i128),
    Big { negative: bool, digits: String },
    Float(f64),
}

/// The exact integer value of a finite, integral float beyond `i128`, as (negative, digits).
fn float_digits(f: f64) -> (bool, String) {
    // Rust formats floats exactly, so this is the float's integer value.
    let s = format!("{:.0}", f.abs());
    (f < 0.0, s)
}

/// Compares two magnitudes given as decimal digits without leading zeros.
fn cmp_digits(a: &str, b: &str) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

fn cmp_big(an: bool, ad: &str, bn: bool, bd: &str) -> Ordering {
    match (an, bn) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        (false, false) => cmp_digits(ad, bd),
        (true, true) => cmp_digits(bd, ad),
    }
}

/// `i` compared with the float `f` exactly (Python compares int and float by value, not by
/// converting the int to a float). `None` when `f` is NaN.
fn cmp_int_float(i: i128, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    const LIMIT: f64 = 1.7014118346046923e38; // 2^127
    if f >= LIMIT {
        return Some(Ordering::Less);
    }
    if f < -LIMIT {
        return Some(Ordering::Greater);
    }
    let t = f.trunc();
    let ti = t as i128; // exact: |t| < 2^127 and integral
    Some(i.cmp(&ti).then(if f > t {
        Ordering::Less
    } else if f < t {
        Ordering::Greater
    } else {
        Ordering::Equal
    }))
}

fn cmp_big_float(negative: bool, digits: &str, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f.is_infinite() {
        return Some(if f > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    // |big| >= 2^127: a float that large is integral, so compare integers.
    let t = f.trunc();
    let (fneg, fdig) = float_digits(t);
    let fneg = fneg && fdig != "0";
    Some(cmp_big(negative, digits, fneg, &fdig).then(if f > t {
        Ordering::Less
    } else if f < t {
        Ordering::Greater
    } else {
        Ordering::Equal
    }))
}

impl Num {
    /// Python's comparison of two numbers; `None` when a NaN is involved (every comparison
    /// with NaN is false).
    pub fn compare(&self, other: &Num) -> Option<Ordering> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) => Some(a.cmp(b)),
            (Num::Float(a), Num::Float(b)) => a.partial_cmp(b),
            (Num::Int(a), Num::Float(b)) => cmp_int_float(*a, *b),
            (Num::Float(a), Num::Int(b)) => cmp_int_float(*b, *a).map(Ordering::reverse),
            (
                Num::Big {
                    negative: an,
                    digits: ad,
                },
                Num::Big {
                    negative: bn,
                    digits: bd,
                },
            ) => Some(cmp_big(*an, ad, *bn, bd)),
            (Num::Big { negative, .. }, Num::Int(_)) => Some(if *negative {
                Ordering::Less
            } else {
                Ordering::Greater
            }),
            (Num::Int(_), Num::Big { negative, .. }) => Some(if *negative {
                Ordering::Greater
            } else {
                Ordering::Less
            }),
            (Num::Big { negative, digits }, Num::Float(f)) => cmp_big_float(*negative, digits, *f),
            (Num::Float(f), Num::Big { negative, digits }) => {
                cmp_big_float(*negative, digits, *f).map(Ordering::reverse)
            }
        }
    }

    fn as_f64(&self) -> PyResult<f64> {
        match self {
            Num::Int(i) => Ok(*i as f64),
            Num::Float(f) => Ok(*f),
            Num::Big { negative, digits } => {
                // float(int) raises OverflowError beyond the float range.
                let f: f64 = digits.parse().unwrap_or(f64::INFINITY);
                if f.is_infinite() {
                    err("OverflowError: int too large to convert to float")
                } else {
                    Ok(if *negative { -f } else { f })
                }
            }
        }
    }

    /// The sign and decimal digits of an integer (`None` for a float).
    fn signed_digits(&self) -> Option<(bool, String)> {
        match self {
            Num::Int(i) => Some((*i < 0, i.unsigned_abs().to_string())),
            Num::Big { negative, digits } => Some((*negative, digits.clone())),
            Num::Float(_) => None,
        }
    }

    /// `self + other`.
    pub fn add(&self, other: &Num) -> PyResult<Num> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) if a.checked_add(*b).is_some() => Ok(Num::Int(a + b)),
            (Num::Float(_), _) | (_, Num::Float(_)) => {
                Ok(Num::Float(self.as_f64()? + other.as_f64()?))
            }
            _ => {
                // Exact integer addition beyond i128, on decimal digits.
                let (an, ad) = self.signed_digits().expect("integer");
                let (bn, bd) = other.signed_digits().expect("integer");
                Ok(if an == bn {
                    int_from_digits(an, &digits_add(&ad, &bd))
                } else if cmp_digits(&ad, &bd) != Ordering::Less {
                    int_from_digits(an, &digits_sub(&ad, &bd))
                } else {
                    int_from_digits(bn, &digits_sub(&bd, &ad))
                })
            }
        }
    }

    /// `self * other`. A product of integers beyond `i128` keeps its sign only (a placeholder
    /// magnitude above `i128`): the one caller compares it with list lengths.
    pub fn mul(&self, other: &Num) -> PyResult<Num> {
        match (self, other) {
            (Num::Int(a), Num::Int(b)) if a.checked_mul(*b).is_some() => Ok(Num::Int(a * b)),
            (Num::Float(_), _) | (_, Num::Float(_)) => {
                Ok(Num::Float(self.as_f64()? * other.as_f64()?))
            }
            _ => {
                let (an, _) = self.signed_digits().expect("integer");
                let (bn, _) = other.signed_digits().expect("integer");
                if matches!(self, Num::Int(0)) || matches!(other, Num::Int(0)) {
                    return Ok(Num::Int(0));
                }
                Ok(Num::Big {
                    negative: an != bn,
                    digits: "1".repeat(40),
                })
            }
        }
    }

    /// `self % other` with Python's sign rule (the result has the divisor's sign).
    pub fn rem(&self, other: &Num) -> PyResult<Num> {
        match (self, other) {
            (Num::Int(_), Num::Int(0)) => err("ZeroDivisionError: integer modulo by zero"),
            (Num::Int(a), Num::Int(b)) => {
                let r = a.checked_rem(*b).unwrap_or(0); // i128::MIN % -1 overflows; it is 0
                Ok(Num::Int(if r != 0 && ((r < 0) != (*b < 0)) {
                    r + b
                } else {
                    r
                }))
            }
            (Num::Big { negative, digits }, Num::Int(m)) => {
                // |a| mod |m| digit by digit, then Python's sign rule.
                let mm = m.unsigned_abs();
                let addmod = |x: u128, y: u128| if x >= mm - y { x - (mm - y) } else { x + y };
                let mut r: u128 = 0;
                for d in digits.bytes() {
                    let r2 = addmod(r, r);
                    let r8 = addmod(addmod(r2, r2), addmod(r2, r2));
                    r = addmod(addmod(r8, r2), u128::from(d - b'0') % mm);
                }
                let r = r as i128; // r < |m| <= 2^127
                Ok(Num::Int(if r == 0 {
                    0
                } else {
                    match (*negative, *m < 0) {
                        (false, false) => r,
                        (true, true) => -r,
                        (false, true) => r + m,
                        (true, false) => m - r,
                    }
                }))
            }
            (Num::Int(a), Num::Big { negative, .. }) => {
                // |a| < |m|: a itself when the signs agree, else a + m.
                if *a == 0 || (*a < 0) == *negative {
                    Ok(Num::Int(*a))
                } else {
                    Num::Int(*a).add(other)
                }
            }
            (Num::Big { .. }, Num::Big { .. }) => {
                err("unsupported: modulo of two integers beyond 128 bits")
            }
            _ => {
                let (a, b) = (self.as_f64()?, other.as_f64()?);
                if b == 0.0 {
                    return err("ZeroDivisionError: float modulo");
                }
                let mut r = a % b;
                if r != 0.0 {
                    if (r < 0.0) != (b < 0.0) {
                        r += b;
                    }
                } else {
                    r = 0.0f64.copysign(b);
                }
                Ok(Num::Float(r))
            }
        }
    }

    /// The hashable key of this number (Python hashes equal numbers equally: `5 == 5.0`).
    pub fn key(&self) -> Key {
        match self {
            Num::Int(i) => Key::Int(*i),
            Num::Big { negative, digits } => Key::Big(*negative, digits.clone()),
            Num::Float(f) if f.is_nan() => Key::NaN,
            Num::Float(f) if f.is_finite() && f.fract() == 0.0 => {
                if f.abs() < 1.7e38 {
                    Key::Int(*f as i128)
                } else {
                    let (n, d) = float_digits(*f);
                    Key::Big(n, d)
                }
            }
            Num::Float(f) => Key::Float(f.to_bits()),
        }
    }
}

/// A hashable Python scalar, normalised so that equal values have equal keys (for the sets of
/// `server.safe_move`). A NaN never equals anything, even itself (the server's NaNs are
/// distinct objects), so sets never report one as a member.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    None,
    Int(i128),
    Big(bool, String),
    Float(u64),
    NaN,
    Str(String),
}

impl PyValue {
    /// `hash`-compatible key; `TypeError` for a list or dict (unhashable).
    pub fn key(&self) -> PyResult<Key> {
        match self {
            PyValue::None => Ok(Key::None),
            PyValue::Str(s) => Ok(Key::Str(s.clone())),
            PyValue::List(_) | PyValue::Dict(_) => err(format!(
                "TypeError: unhashable type: '{}'",
                self.type_name()
            )),
            other => Ok(other.num().expect("numeric").key()),
        }
    }
}

/// Deepest comparison the server's `==` / `<` survives (Python's recursion limit).
const MAX_COMPARE_DEPTH: usize = 900;

/// Python's `a == b`. `RecursionError` beyond 900 nested containers.
pub fn py_eq(a: &PyValue, b: &PyValue) -> PyResult<bool> {
    eq_depth(a, b, 0)
}

fn eq_depth(a: &PyValue, b: &PyValue, depth: usize) -> PyResult<bool> {
    if depth > MAX_COMPARE_DEPTH {
        return err("RecursionError: maximum recursion depth exceeded in comparison");
    }
    Ok(match (a, b) {
        (PyValue::None, PyValue::None) => true,
        (PyValue::Str(x), PyValue::Str(y)) => x == y,
        (PyValue::List(x), PyValue::List(y)) => {
            if x.len() != y.len() {
                return Ok(false);
            }
            for (p, q) in x.iter().zip(y) {
                if !eq_depth(p, q, depth + 1)? {
                    return Ok(false);
                }
            }
            true
        }
        (PyValue::Dict(x), PyValue::Dict(y)) => {
            if x.len() != y.len() {
                return Ok(false);
            }
            for (k, v) in x {
                match b.get(k) {
                    Some(w) if eq_depth(v, w, depth + 1)? => {}
                    _ => return Ok(false),
                }
            }
            true
        }
        _ => match (a.num(), b.num()) {
            (Some(x), Some(y)) => x.compare(&y) == Some(Ordering::Equal),
            _ => false,
        },
    })
}

/// Python's ordering of `a` and `b` for `<`, `<=`, `>`, `>=`: numbers by value, strings by
/// code point, lists lexicographically. `Ok(None)` when a NaN makes every comparison false;
/// `TypeError` for other type pairs.
pub fn py_cmp(a: &PyValue, b: &PyValue) -> PyResult<Option<Ordering>> {
    cmp_depth(a, b, 0)
}

fn cmp_depth(a: &PyValue, b: &PyValue, depth: usize) -> PyResult<Option<Ordering>> {
    if depth > MAX_COMPARE_DEPTH {
        return err("RecursionError: maximum recursion depth exceeded in comparison");
    }
    match (a, b) {
        (PyValue::Str(x), PyValue::Str(y)) => Ok(Some(x.cmp(y))),
        (PyValue::List(x), PyValue::List(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !eq_depth(p, q, depth + 1)? {
                    return cmp_depth(p, q, depth + 1);
                }
            }
            Ok(Some(x.len().cmp(&y.len())))
        }
        _ => match (a.num(), b.num()) {
            (Some(x), Some(y)) => Ok(x.compare(&y)),
            _ => err(format!(
                "TypeError: '<' not supported between instances of '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )),
        },
    }
}

/// `a >= b`.
pub fn py_ge(a: &PyValue, b: &PyValue) -> PyResult<bool> {
    Ok(matches!(
        py_cmp(a, b)?,
        Some(Ordering::Greater | Ordering::Equal)
    ))
}

/// `lo <= x < hi` for a number `x` and integer bounds, evaluated as Python does (the second
/// comparison only when the first holds). `TypeError` when `x` is not a number.
pub fn in_range(x: &PyValue, lo: i128, hi: i128) -> PyResult<bool> {
    let Some(n) = x.num() else {
        return err(format!(
            "TypeError: '<=' not supported between instances of 'int' and '{}'",
            x.type_name()
        ));
    };
    Ok(Num::Int(lo)
        .compare(&n)
        .is_some_and(|o| o != Ordering::Greater)
        && n.compare(&Num::Int(hi)) == Some(Ordering::Less))
}

/// The number of `x` for arithmetic, or `TypeError`.
pub fn number(x: &PyValue) -> PyResult<Num> {
    x.num().map_or_else(
        || {
            err(format!(
                "TypeError: unsupported operand type for arithmetic: '{}'",
                x.type_name()
            ))
        },
        Ok,
    )
}

// --- int() and numpy's int32 store ---------------------------------------------------------------

/// Python's limit on the digits of an integer read from a string (`sys.int_info`), which
/// `json.loads` and `int()` enforce.
pub const MAX_STR_DIGITS: usize = 4300;

/// The first code point of each run of ten Unicode decimal digits (category Nd), from Python's
/// `unicodedata` 15.0 (the venv's), so `int()` reads `"\u{661}\u{662}"` as 12 as Python does.
const DIGIT_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// The decimal value of a Unicode decimal digit (`unicodedata.decimal`).
fn decimal_digit(c: char) -> Option<u8> {
    let cp = c as u32;
    let i = DIGIT_ZEROS.partition_point(|&z| z <= cp);
    let zero = *DIGIT_ZEROS.get(i.checked_sub(1)?)?;
    (cp - zero < 10).then(|| (cp - zero) as u8)
}

/// `int(text)` in base 10: surrounding whitespace, an optional sign, decimal digits of any
/// script with single underscores between them, at most [`MAX_STR_DIGITS`] digits. `None`
/// where Python raises `ValueError`.
pub fn python_int(text: &str) -> Option<Num> {
    let t = text.trim();
    let (negative, body) = match t.chars().next()? {
        '-' => (true, &t[1..]),
        '+' => (false, &t[1..]),
        _ => (false, t),
    };
    let mut digits = String::new();
    let mut last_underscore = true; // no leading underscore
    for c in body.chars() {
        if c == '_' {
            if last_underscore {
                return None;
            }
            last_underscore = true;
        } else {
            digits.push(char::from(b'0' + decimal_digit(c)?));
            last_underscore = false;
        }
    }
    if digits.is_empty() || last_underscore || digits.len() > MAX_STR_DIGITS {
        return None;
    }
    Some(int_from_digits(negative, &digits))
}

/// The sum of two magnitudes in decimal digits.
fn digits_add(a: &str, b: &str) -> String {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let (mut i, mut j, mut carry) = (a.len(), b.len(), 0u8);
    while i > 0 || j > 0 || carry > 0 {
        let mut d = carry;
        if i > 0 {
            i -= 1;
            d += a[i] - b'0';
        }
        if j > 0 {
            j -= 1;
            d += b[j] - b'0';
        }
        out.push(b'0' + d % 10);
        carry = d / 10;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII digits")
}

/// `a - b` for magnitudes in decimal digits with `a >= b`.
fn digits_sub(a: &str, b: &str) -> String {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut out = Vec::with_capacity(a.len());
    let (mut j, mut borrow) = (b.len(), 0i8);
    for i in (0..a.len()).rev() {
        let mut d = (a[i] - b'0') as i8 - borrow;
        if j > 0 {
            j -= 1;
            d -= (b[j] - b'0') as i8;
        }
        borrow = i8::from(d < 0);
        out.push(b'0' + d.rem_euclid(10) as u8);
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII digits")
}

/// An integer from its sign and ASCII decimal digits.
fn int_from_digits(negative: bool, digits: &str) -> Num {
    let trimmed = digits.trim_start_matches('0');
    let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
    match trimmed.parse::<i128>() {
        Ok(v) => Num::Int(if negative { -v } else { v }),
        Err(_) => Num::Big {
            negative,
            digits: trimmed.to_string(),
        },
    }
}

/// numpy's store of a Python object into an `int32` array (`health[i] = s["health"]`,
/// `head[i] = segs[0]`, `jnp.asarray(turn, jnp.int32)`; engine_json.py:113-116, 146): a bool
/// is 0 or 1, a float is truncated (NaN and infinities raise), a string goes through `int()`,
/// and anything outside `int32` raises `OverflowError`. Measured on numpy 2.5 with every
/// kind of input (see the api crate's tests).
pub fn int32(v: &PyValue) -> PyResult<i32> {
    let n = match v {
        PyValue::Bool(b) => Num::Int(i128::from(*b)),
        PyValue::Int(i) => Num::Int(*i),
        PyValue::BigInt { .. } => return err("OverflowError: Python int too large for int32"),
        PyValue::Float(f) if f.is_nan() => {
            return err("ValueError: cannot convert float NaN to integer");
        }
        PyValue::Float(f) if f.is_infinite() => {
            return err("OverflowError: cannot convert float infinity to integer");
        }
        PyValue::Float(f) => {
            let t = f.trunc();
            if t.abs() >= 1e30 {
                return err("OverflowError: Python int too large for int32");
            }
            Num::Int(t as i128)
        }
        PyValue::Str(s) => match python_int(s) {
            Some(n) => n,
            None => return err(format!("ValueError: invalid literal for int(): {s:?}")),
        },
        other => {
            return err(format!(
                "TypeError: int() argument must be a string or a number, not '{}'",
                other.type_name()
            ));
        }
    };
    match n {
        Num::Int(i) => i32::try_from(i).map_or_else(
            |_| err(format!("OverflowError: {i} out of bounds for int32")),
            Ok,
        ),
        _ => err("OverflowError: Python int too large for int32"),
    }
}

/// A numpy integer index into an axis of `size`, as `food[y, x]` reads and writes one cell: a
/// Python `int` in `[-size, size)`, negative ones counted from the end. A list holding exactly
/// one such `int` (at any nesting) selects the same single cell. `IndexError` / `ValueError`
/// otherwise: a float or a string is not an index, a longer or empty list selects several or
/// no cells (whose truth value `if food[y, x]` cannot take), and a bool is a mask (also
/// several or no cells).
pub fn numpy_index(v: &PyValue, size: usize) -> PyResult<usize> {
    let mut v = v;
    for _ in 0..64 {
        match v {
            PyValue::List(items) if items.len() == 1 => v = &items[0],
            _ => break,
        }
    }
    let size = size as i128;
    match v {
        PyValue::Int(i) if (-size..size).contains(i) => {
            Ok(if *i < 0 { i + size } else { *i } as usize)
        }
        PyValue::Int(i) => err(format!(
            "IndexError: index {i} is out of bounds for axis with size {size}"
        )),
        other => err(format!(
            "IndexError: not a single-cell index: '{}'",
            other.type_name()
        )),
    }
}

// --- Parsing -----------------------------------------------------------------------------------

/// Deepest nesting of arrays and objects `json.loads` accepts (Python 3.12's C recursion limit,
/// measured: 9997 nested containers parse, 9998 raise `RecursionError`).
pub const MAX_DEPTH: usize = 9997;

enum Frame {
    List(Vec<PyValue>),
    Dict {
        items: Vec<(String, PyValue)>,
        /// Key -> position, built once the object is large enough to need it.
        index: Option<HashMap<String, usize>>,
        key: String,
    },
}

impl Frame {
    fn depth_check(depth: usize) -> Result<(), String> {
        if depth > MAX_DEPTH {
            Err("RecursionError: maximum recursion depth exceeded while decoding JSON".into())
        } else {
            Ok(())
        }
    }
}

fn insert(
    items: &mut Vec<(String, PyValue)>,
    index: &mut Option<HashMap<String, usize>>,
    key: String,
    value: PyValue,
) {
    if index.is_none() && items.len() >= 16 {
        *index = Some(
            items
                .iter()
                .enumerate()
                .map(|(i, (k, _))| (k.clone(), i))
                .collect(),
        );
    }
    let pos = match index {
        Some(map) => map.get(&key).copied(),
        None => items.iter().position(|(k, _)| *k == key),
    };
    match pos {
        Some(p) => items[p].1 = value,
        None => {
            if let Some(map) = index {
                map.insert(key.clone(), items.len());
            }
            items.push((key, value));
        }
    }
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        i += 1;
    }
    i
}

/// `json.loads(text)` (module docs). The error is the reason, `JSONDecodeError`-like.
pub fn parse(text: &str) -> Result<PyValue, String> {
    let b = text.as_bytes();
    let mut i = skip_ws(b, 0);
    let mut stack: Vec<Frame> = Vec::new();
    loop {
        // A value starts at `i`: a scalar, or a container whose first item comes next.
        let mut value = match b.get(i) {
            Some(b'[') => {
                i = skip_ws(b, i + 1);
                if b.get(i) == Some(&b']') {
                    i += 1;
                    PyValue::List(Vec::new())
                } else {
                    Frame::depth_check(stack.len() + 1)?;
                    stack.push(Frame::List(Vec::new()));
                    continue;
                }
            }
            Some(b'{') => {
                i = skip_ws(b, i + 1);
                if b.get(i) == Some(&b'}') {
                    i += 1;
                    PyValue::Dict(Vec::new())
                } else {
                    Frame::depth_check(stack.len() + 1)?;
                    let key;
                    (key, i) = parse_key(b, i)?;
                    stack.push(Frame::Dict {
                        items: Vec::new(),
                        index: None,
                        key,
                    });
                    continue;
                }
            }
            _ => {
                let (v, next) = parse_scalar(text, i)?;
                i = next;
                v
            }
        };
        // Attach the finished value to its parents, closing every container that ends here.
        loop {
            let Some(top) = stack.last_mut() else {
                i = skip_ws(b, i);
                if i != b.len() {
                    return Err(format!("Extra data at byte {i}"));
                }
                return Ok(value);
            };
            match top {
                Frame::List(items) => {
                    items.push(value);
                    i = skip_ws(b, i);
                    match b.get(i) {
                        Some(b',') => {
                            i = skip_ws(b, i + 1);
                            break;
                        }
                        Some(b']') => {
                            i += 1;
                            let Some(Frame::List(items)) = stack.pop() else {
                                unreachable!("top frame is a list")
                            };
                            value = PyValue::List(items);
                        }
                        _ => return Err(format!("Expecting ',' delimiter at byte {i}")),
                    }
                }
                Frame::Dict { items, index, key } => {
                    insert(items, index, std::mem::take(key), value);
                    i = skip_ws(b, i);
                    match b.get(i) {
                        Some(b',') => {
                            i = skip_ws(b, i + 1);
                            let next;
                            (next, i) = parse_key(b, i)?;
                            *key = next;
                            break;
                        }
                        Some(b'}') => {
                            i += 1;
                            let Some(Frame::Dict { items, .. }) = stack.pop() else {
                                unreachable!("top frame is a dict")
                            };
                            value = PyValue::Dict(items);
                        }
                        _ => return Err(format!("Expecting ',' delimiter at byte {i}")),
                    }
                }
            }
        }
    }
}

/// An object key and the `:` after it; returns the key and the index of its value.
fn parse_key(b: &[u8], i: usize) -> Result<(String, usize), String> {
    if b.get(i) != Some(&b'"') {
        return Err(format!(
            "Expecting property name enclosed in double quotes at byte {i}"
        ));
    }
    let (key, i) = parse_string(b, i + 1)?;
    let i = skip_ws(b, i);
    if b.get(i) != Some(&b':') {
        return Err(format!("Expecting ':' delimiter at byte {i}"));
    }
    Ok((key, skip_ws(b, i + 1)))
}

/// A string whose opening quote is just before `i`; returns it and the index after the
/// closing quote.
fn parse_string(b: &[u8], mut i: usize) -> Result<(String, usize), String> {
    let mut out: Vec<u8> = Vec::new();
    loop {
        let start = i;
        while let Some(&c) = b.get(i) {
            if c == b'"' || c == b'\\' || c < 0x20 {
                break;
            }
            i += 1;
        }
        out.extend_from_slice(&b[start..i]);
        match b.get(i) {
            None => return Err("Unterminated string".into()),
            Some(b'"') => {
                // The input is UTF-8 and escapes push whole characters, so this cannot fail.
                let s = String::from_utf8(out).map_err(|e| e.to_string())?;
                return Ok((s, i + 1));
            }
            Some(b'\\') => {
                let esc = *b.get(i + 1).ok_or("Unterminated string")?;
                i += 2;
                let ch = match esc {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let hi = hex4(b, i)?;
                        i += 4;
                        let code = if (0xD800..0xDC00).contains(&hi)
                            && b.get(i) == Some(&b'\\')
                            && b.get(i + 1) == Some(&b'u')
                        {
                            match hex4(b, i + 2) {
                                Ok(lo) if (0xDC00..0xE000).contains(&lo) => {
                                    i += 6;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                }
                                _ => hi,
                            }
                        } else {
                            hi
                        };
                        // A lone surrogate (Python keeps it) cannot be in a Rust string.
                        char::from_u32(code).unwrap_or('\u{FFFD}')
                    }
                    _ => return Err(format!("Invalid \\escape at byte {}", i - 2)),
                };
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
            Some(_) => return Err(format!("Invalid control character at byte {i}")),
        }
    }
}

fn hex4(b: &[u8], i: usize) -> Result<u32, String> {
    let s = b.get(i..i + 4).ok_or("Invalid \\uXXXX escape")?;
    let s = std::str::from_utf8(s).map_err(|_| "Invalid \\uXXXX escape")?;
    if !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("Invalid \\uXXXX escape".into());
    }
    u32::from_str_radix(s, 16).map_err(|_| "Invalid \\uXXXX escape".into())
}

/// A scalar at `i`: a string, a number, or a literal.
fn parse_scalar(text: &str, i: usize) -> Result<(PyValue, usize), String> {
    let b = text.as_bytes();
    let rest = &b[i.min(b.len())..];
    let literal = |word: &str, v: PyValue| -> Option<(PyValue, usize)> {
        rest.starts_with(word.as_bytes())
            .then(|| (v, i + word.len()))
    };
    let found = match rest.first() {
        Some(b'"') => return parse_string(b, i + 1).map(|(s, j)| (PyValue::Str(s), j)),
        Some(b'n') => literal("null", PyValue::None),
        Some(b't') => literal("true", PyValue::Bool(true)),
        Some(b'f') => literal("false", PyValue::Bool(false)),
        Some(b'N') => literal("NaN", PyValue::Float(f64::NAN)),
        Some(b'I') => literal("Infinity", PyValue::Float(f64::INFINITY)),
        Some(b'-') if rest.starts_with(b"-Infinity") => {
            Some((PyValue::Float(f64::NEG_INFINITY), i + 9))
        }
        Some(b'-' | b'0'..=b'9') => parse_number(text, i),
        _ => None,
    };
    found.map_or_else(|| Err(format!("Expecting value at byte {i}")), Ok)
}

/// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?` (`json.scanner.NUMBER_RE`): an `int` without a
/// fraction or exponent, else a `float`. A fraction or exponent without digits is not part of
/// the number (it is then extra data).
fn parse_number(text: &str, start: usize) -> Option<(PyValue, usize)> {
    let b = text.as_bytes();
    let digits_from = |mut j: usize| {
        let s = j;
        while b.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        (j, j - s)
    };
    let mut i = start;
    let negative = b.get(i) == Some(&b'-');
    if negative {
        i += 1;
    }
    let int_start = i;
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits_from(i).0,
        _ => return None,
    }
    let int_end = i;
    let mut is_float = false;
    if b.get(i) == Some(&b'.') {
        let (j, n) = digits_from(i + 1);
        if n > 0 {
            i = j;
            is_float = true;
        }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(b.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let (k, n) = digits_from(j);
        if n > 0 {
            i = k;
            is_float = true;
        }
    }
    if is_float {
        let f: f64 = text[start..i].parse().ok()?;
        return Some((PyValue::Float(f), i));
    }
    let digits = &text[int_start..int_end];
    if digits.len() > MAX_STR_DIGITS {
        return None; // ValueError: exceeds the limit for integer string conversion
    }
    let v = match int_from_digits(negative, digits) {
        Num::Int(v) => PyValue::Int(v),
        Num::Big { negative, digits } => PyValue::BigInt { negative, digits },
        Num::Float(_) => unreachable!("integers only"),
    };
    Some((v, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(text: &str) -> Result<PyValue, String> {
        parse(text)
    }

    #[test]
    fn python_json_dialect() {
        assert_eq!(p("1").unwrap(), PyValue::Int(1));
        assert_eq!(p("-0").unwrap(), PyValue::Int(0));
        assert_eq!(p(" [1] \n").unwrap(), PyValue::List(vec![PyValue::Int(1)]));
        assert_eq!(p("1e5").unwrap(), PyValue::Float(1e5));
        assert_eq!(p("1E+2").unwrap(), PyValue::Float(100.0));
        assert!(matches!(p("NaN").unwrap(), PyValue::Float(f) if f.is_nan()));
        assert_eq!(p("-Infinity").unwrap(), PyValue::Float(f64::NEG_INFINITY));
        assert_eq!(p("1e400").unwrap(), PyValue::Float(f64::INFINITY));
        assert_eq!(
            p(r#""😀\ud800""#).unwrap(),
            PyValue::Str("\u{1F600}\u{FFFD}".into())
        );
        assert_eq!(p("\"\u{7f}\"").unwrap(), PyValue::Str("\u{7f}".into()));
        let d = p(r#"{"a":1,"b":2,"a":3}"#).unwrap();
        assert_eq!(
            d,
            PyValue::Dict(vec![
                ("a".into(), PyValue::Int(3)),
                ("b".into(), PyValue::Int(2))
            ])
        );
        let big = format!("-1{}", "0".repeat(399));
        assert_eq!(
            p(&big).unwrap(),
            PyValue::BigInt {
                negative: true,
                digits: format!("1{}", "0".repeat(399))
            }
        );
        assert!(p(&format!("1{}", "0".repeat(4299))).is_ok());
        for bad in [
            "",
            "01",
            "1.",
            "-",
            "+1",
            "-NaN",
            "nan",
            "[1,]",
            "{\"a\":1,}",
            "\"a\tb\"",
            r#""\x""#,
            "[1]\u{c}",
            "[1]\u{0}",
            "{1:2}",
            "[1 2]",
            "nul",
            "\"abc",
            r#""\u12""#,
        ] {
            assert!(p(bad).is_err(), "{bad:?}");
        }
        assert!(p(&format!("1{}", "0".repeat(4300))).is_err());
    }

    #[test]
    fn nesting_limit_and_iterative_drop() {
        let ok = format!("{}1{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        drop(p(&ok).unwrap());
        let deep = MAX_DEPTH + 1;
        assert!(p(&format!("{}1{}", "[".repeat(deep), "]".repeat(deep))).is_err());
        let objs = format!("{}1{}", "{\"a\":".repeat(MAX_DEPTH), "}".repeat(MAX_DEPTH));
        drop(p(&objs).unwrap());
        // A million-deep value (built directly) drops without overflowing the stack.
        let mut v = PyValue::Int(0);
        for _ in 0..1_000_000 {
            v = PyValue::List(vec![v]);
        }
        drop(v);
    }

    #[test]
    fn large_objects_keep_the_last_duplicate() {
        let mut text = String::from("{");
        for k in 0..100 {
            text.push_str(&format!("\"k{k}\":{k},"));
        }
        text.push_str("\"k3\":-1}");
        let v = p(&text).unwrap();
        assert_eq!(v.get("k3"), Some(&PyValue::Int(-1)));
        assert_eq!(v.len().unwrap(), 100);
    }

    #[test]
    fn int_conversions_like_python() {
        let int = |s: &str| python_int(s);
        assert_eq!(int(" +1_2 "), Some(Num::Int(12)));
        assert_eq!(int("\u{661}\u{662}"), Some(Num::Int(12)));
        assert_eq!(int("\u{FF11}\u{FF12}"), Some(Num::Int(12)));
        assert_eq!(int("\u{3000}12\u{2028}"), Some(Num::Int(12)));
        assert_eq!(int("007"), Some(Num::Int(7)));
        for bad in [
            "\u{1c}12",
            "\u{200b}12",
            "1__2",
            "_1",
            "1_",
            "",
            "-",
            "3.5",
            "0x10",
            "+-1",
        ] {
            assert_eq!(int(bad), None, "{bad:?}");
        }
        assert!(matches!(int(&"9".repeat(40)), Some(Num::Big { .. })));
        assert_eq!(int(&"9".repeat(4301)), None);
        // numpy's int32 store (probed on numpy 2.5).
        let i32of = |v: PyValue| int32(&v).ok();
        assert_eq!(i32of(PyValue::Float(89.7)), Some(89));
        assert_eq!(i32of(PyValue::Float(-89.7)), Some(-89));
        assert_eq!(i32of(PyValue::Float(2147483647.9)), Some(i32::MAX));
        assert_eq!(i32of(PyValue::Float(2147483648.0)), None);
        assert_eq!(i32of(PyValue::Float(f64::NAN)), None);
        assert_eq!(i32of(PyValue::Float(f64::INFINITY)), None);
        assert_eq!(i32of(PyValue::Str(" 9_0 ".into())), Some(90));
        assert_eq!(i32of(PyValue::Str("90.0".into())), None);
        assert_eq!(i32of(PyValue::Str("2147483648".into())), None);
        assert_eq!(i32of(PyValue::Bool(true)), Some(1));
        assert_eq!(i32of(PyValue::None), None);
        assert_eq!(i32of(PyValue::List(vec![PyValue::Int(5)])), None);
        assert_eq!(i32of(PyValue::Int(-(1 << 31))), Some(i32::MIN));
        assert_eq!(i32of(PyValue::Int(1 << 31)), None);
    }

    #[test]
    fn python_comparisons() {
        let (one, onef, t) = (PyValue::Int(1), PyValue::Float(1.0), PyValue::Bool(true));
        assert!(py_eq(&one, &onef).unwrap() && py_eq(&one, &t).unwrap());
        assert!(!py_eq(&PyValue::Str("1".into()), &one).unwrap());
        let nan = PyValue::Float(f64::NAN);
        assert!(!py_eq(&nan, &nan).unwrap());
        assert_eq!(py_cmp(&nan, &one).unwrap(), None);
        assert!(py_cmp(&PyValue::Str("9".into()), &one).is_err());
        assert!(py_ge(&PyValue::Str("9".into()), &PyValue::Str("10".into())).unwrap());
        // Exact int/float comparison: 2^53 + 1 > 2^53 as a float.
        let big = PyValue::Int((1 << 53) + 1);
        assert_eq!(
            py_cmp(&big, &PyValue::Float(9007199254740992.0)).unwrap(),
            Some(Ordering::Greater)
        );
        let huge = parse(&"9".repeat(50)).unwrap();
        assert_eq!(
            py_cmp(&huge, &PyValue::Float(1e49)).unwrap(),
            Some(Ordering::Greater)
        );
        assert_eq!(
            py_cmp(&huge, &PyValue::Float(1e50)).unwrap(),
            Some(Ordering::Less)
        );
        assert_eq!(
            py_cmp(&huge, &PyValue::Int(i128::MAX)).unwrap(),
            Some(Ordering::Greater)
        );
        assert!(in_range(&PyValue::Float(10.5), 0, 11).unwrap());
        assert!(!in_range(&PyValue::Int(-1), 0, 11).unwrap());
        assert!(in_range(&PyValue::None, 0, 11).is_err());
        // Keys: 5 == 5.0 == True-ish hashing.
        assert_eq!(
            PyValue::Float(5.0).key().unwrap(),
            PyValue::Int(5).key().unwrap()
        );
        assert_eq!(PyValue::Float(-0.0).key().unwrap(), Key::Int(0));
        assert!(PyValue::List(vec![]).key().is_err());
    }

    #[test]
    fn big_integer_arithmetic() {
        let big = |s: &str| match parse(s).unwrap().num().unwrap() {
            n @ Num::Big { .. } => n,
            other => panic!("not big: {other:?}"),
        };
        let p40 = big(&format!("1{}", "0".repeat(40)));
        let sum = p40.add(&Num::Int(-1)).unwrap();
        assert_eq!(
            sum,
            Num::Big {
                negative: false,
                digits: "9".repeat(40)
            }
        );
        assert_eq!(sum.add(&Num::Int(1)).unwrap(), p40);
        let neg = big(&format!("-1{}", "0".repeat(40)));
        assert_eq!(neg.add(&p40).unwrap(), Num::Int(0));
        // Back into i128.
        let just_over = Num::Int(i128::MAX).add(&Num::Int(1)).unwrap();
        assert!(matches!(just_over, Num::Big { .. }));
        assert_eq!(just_over.add(&Num::Int(-1)).unwrap(), Num::Int(i128::MAX));
        // 10^40 mod 11 = 1 (10 = -1 mod 11, even power); Python's sign rule.
        assert_eq!(p40.rem(&Num::Int(11)).unwrap(), Num::Int(1));
        assert_eq!(neg.rem(&Num::Int(11)).unwrap(), Num::Int(10));
        assert_eq!(p40.rem(&Num::Int(-11)).unwrap(), Num::Int(-10));
        assert_eq!(neg.rem(&Num::Int(-11)).unwrap(), Num::Int(-1));
        // 10^40 mod (2^127 - 1), from Python.
        assert_eq!(
            p40.rem(&Num::Int(i128::MAX)).unwrap(),
            Num::Int(131_811_359_292_784_559_562_136_384_478_721_867_834)
        );
        assert_eq!(Num::Int(5).rem(&p40).unwrap(), Num::Int(5));
        assert_eq!(
            Num::Int(-1).rem(&p40).unwrap(),
            Num::Big {
                negative: false,
                digits: "9".repeat(40)
            }
        );
        assert!(p40.rem(&p40).is_err());
    }

    #[test]
    fn python_modulo() {
        let m = |a: Num, b: Num| a.rem(&b).unwrap();
        assert_eq!(m(Num::Int(-1), Num::Int(11)), Num::Int(10));
        assert_eq!(m(Num::Int(1), Num::Int(-11)), Num::Int(-10));
        assert_eq!(m(Num::Float(-1.0), Num::Int(11)), Num::Float(10.0));
        assert!(Num::Int(1).rem(&Num::Int(0)).is_err());
        assert!(Num::Float(1.0).rem(&Num::Float(0.0)).is_err());
        assert_eq!(
            m(Num::Int(i128::from(i64::MAX) + 1), Num::Int(11)),
            Num::Int((i128::from(i64::MAX) + 1) % 11)
        );
    }
}
