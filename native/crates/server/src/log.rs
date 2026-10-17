//! Log lines in the Python server's format (`logging.basicConfig(format="%(asctime)s
//! %(message)s")`): local time with milliseconds, then the message, on stderr.

use std::fmt::Arguments;
use std::sync::atomic::{AtomicBool, Ordering};

static QUIET: AtomicBool = AtomicBool::new(false);

/// Silences all log lines (tests).
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// The timestamp Python's logging prints: `2026-10-15 18:32:19,883`.
pub fn timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S,%3f")
        .to_string()
}

/// Writes one log line (use the [`crate::info!`] macro).
pub fn line(args: Arguments<'_>) {
    if !QUIET.load(Ordering::Relaxed) {
        eprintln!("{} {}", timestamp(), args);
    }
}

/// Logs a formatted line: `info!("{} game {}", snake, game)`.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => { $crate::log::line(format_args!($($arg)*)) };
}
