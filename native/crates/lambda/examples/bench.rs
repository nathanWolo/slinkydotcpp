//! The Lambda's `{"bench": ...}` invocation run locally (see `slinky_lambda::bench`):
//!
//! ```text
//! cargo run --release -p slinky-lambda --example bench -- <run dir> ['{"positions": 12}']
//! ```

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let spec = args.get(2).map_or(serde_json::json!({}), |s| {
        serde_json::from_str(s).expect("a JSON spec")
    });
    let out = slinky_lambda::bench::run(std::path::Path::new(&args[1]), &spec, None);
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
